import time
import numpy as np
import torch
import time
import os
import warnings
import math
import onnxruntime
from typing import List, Tuple, Dict, Union
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import OrderedDict
warnings.filterwarnings("ignore")
from commons import LlmBaseModel

HEAD_NUM = 32
GRAPH_SCHE_DESC = 1
GRAPH_CDMA_DESC = 2
GRAPH_CDMA_DATA = 3
GRAPH_KERL_FUNC = 4
GRAPH_KERL_PARA = 5

UPDATE_HEAD = (0x1 << 5) 
UPDATE_TAIL = (0x1 << 6) 
UPDATE_INTE = (0x1 << 7)
LOAD_SHAR = (0x1 << 8) 
LOAD_UNIQ = (0x1 << 9)

class Llama3Model(LlmBaseModel):
    r"""
    @class Llama3Model
    @brief Inherit LlmBaseModel and implement the inference function of llama3.
    """
    def __init__(self, 
                 rpp_dir: str,
                 input_size: int,
                 target_len: int,
                 write_file: int=0,
                 low_power: int=0,
                 prefix: int=1,
                 perf_mode: int=1,
                 print_tokens: int=1):
        r"""
        Class init function, construct the Llama3Model class and initialize the parameters.
        
        Args:
            rpp_dir (`str`):
                The graph model path
            low_power (`int`):
                The flag of low power, 0-not low power, 1-low power
            input_size (`int`): 
                The maximum input size supported by the model
            target_len (`int`):
                The maximum total size supported by the model
            write_file (`int`): 
                The flag instruct to write the output result to the file, 1-write file, 0-not write
            prefix (`int`):
                The prefix of prompt
            print_tokens (`int`):
                Whether to print the data flag, 1-print, 0-not
        """
        super(Llama3Model, self).__init__(rpp_dir=rpp_dir,
                                          graph_engine=None, 
                                          low_power=low_power,
                                          input_size=input_size,
                                          target_len=target_len)
        self.write_file = write_file
        self.perf_mode = perf_mode
        self.prefix = prefix
        self.print_tokens = print_tokens
        
        self.dtype = torch.float32
        self.provider = "CPUExecutionProvider"
        self.decoding_mask_num = (-1 * math.exp(60))
        self.attention_mask = np.ones((1, 1, 1, self.target_len), dtype=np.float32)     
        self.prompt_len = -1

        self.hidden_states_bindings   = None
        self.attention_mask_bindings  = None
        self.decode_host_buffer_list  = [] #save decode lm_head output
        self.prefill_host_buffer_list = [] #save prefill lm_head output

        self.cached_token = []
        if self.write_file:
            self.write_file = open("prompt_result.txt", "w")
        if self.low_power:
            self._graph_set_power_mode(1)
    
    def _set_sub_graph_names(self):
        r"""
        Set the sub names for graph models
        """
        # prefill
        self.prefill_top_name = 'loop1_decoders/top'
        self.prefill_mid_name = 'loop1_decoders/mid'
        self.prefill_bot_name = 'loop1_decoders/bot'
        self.prefill_rms_norm_name = 'loop1_others/rms_norms/rms_norm'
        self.prefill_rms_param_name = 'loop1_others/rms_norms/rms_params'
        self.prefill_lm_head_name  = 'loop1_others/lm_heads/lm_head'
        self.prefill_top_param_name = 'loop1_others/top_params'
        self.prefill_mid_param_name = 'loop1_others/mid_params'
        self.prefill_bot_param_name = 'loop1_others/bot_params'
        self.prefill_k_quant_name = 'loop1_others/k_quants/k_quant'
        self.prefill_v_quant_name = 'loop1_others/v_quants/v_quant'
        self.prefill_embed_name = 'loop1_others/embed_tokens'
        self.prefill_k_reformat_name = 'loop1_others/kv_reformats/k_reformat'
        self.prefill_v_reformat_name = 'loop1_others/kv_reformats/v_reformat'

        # 1oop2
        self.decode_decodes_name  = 'loop2_decoders/decoder'
        self.decode_lm_head_name  = 'loop2_others/lm_heads/lm_head'
        self.decode_rms_norm_name = 'loop2_others/rms_norm'
        self.decode_kv_name    = 'loop2_others/kv_plugin'
        self.decode_param_name = 'loop2_others/graph_params/graph_param_'

    def _allocate_buffers(self):
        r"""
        Get bindings and allocate the host buffers for graph model
        """
        self.global_share_id = {
            self.prefill_top_param_name:self._share_index_increment(),
            self.prefill_mid_param_name:self._share_index_increment(),
            self.prefill_bot_param_name:self._share_index_increment(),
            self.prefill_rms_param_name:self._share_index_increment(),
            self.decode_param_name:self._share_index_increment(),
            self.prefill_k_quant_name:self._share_index_increment(),
            self.prefill_v_quant_name:self._share_index_increment(),
            self.prefill_k_reformat_name:self._share_index_increment(),
            self.prefill_v_reformat_name:self._share_index_increment()
        }

        ## decode
        graph_decoders_x: str = f'{self.decode_decodes_name}{0}'
        self.decode_decoder_i_bindings = self._alloc_graph_io_params(graph_decoders_x, 1)
        
        self.decode_kv_i_bindings = self._alloc_graph_io_params(self.decode_kv_name, 1)

        self.decode_lm_heads_o_bindings_list = [] 
        for i in range(0, 4):
            tmp_lm_head: str = f'{self.decode_lm_head_name}{i}'
            lm_head_out_bindings = self._alloc_graph_io_params(tmp_lm_head, 2)
            self.decode_lm_heads_o_bindings_list.append(lm_head_out_bindings)

        mul_ratio = self.input_size // 128
        
        ## prefill
        # prefill top in
        graph_decoders_top0: str = f'{self.prefill_top_name}0'
        self.prefill_decoder_top_i_0bindings = OrderedDict()
        prefill_decoder_top_i_0binding_0 = self.graph_engine.get_rpp_fw_in_one(graph_decoders_top0,
                                                                                '/self_attn/Unsqueeze_output_0',
                                                                                1*1*128*128*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_top_i_0bindings['/self_attn/Unsqueeze_output_0'] = prefill_decoder_top_i_0binding_0
        prefill_decoder_top_i_0binding_1 = self.graph_engine.get_rpp_fw_in_one(graph_decoders_top0,
                                                                                '/self_attn/Unsqueeze_1_output_0',
                                                                                1*1*128*128*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_top_i_0bindings['/self_attn/Unsqueeze_1_output_0'] = prefill_decoder_top_i_0binding_1
        prefill_decoder_top_i_0binding_2 = self.graph_engine.get_rpp_fw_in_one(graph_decoders_top0,
                                                                                'hidden_in',
                                                                                1*128*4096*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_top_i_0bindings['hidden_in'] = prefill_decoder_top_i_0binding_2
        # prefill top out
        # self.prefill_decoder_top_o_0bindings = OrderedDict()
        # graph_decoders_topx: str = f'{self.prefill_top_name}{0}'
        # prefill_decoder_top_o_binding_k = self.graph_engine.get_rpp_fw_out_one(graph_decoders_topx,
        #                                                                     'past_key0',
        #                                                                     1*8*128*128*2*mul_ratio,
        #                                                                     self.IO_DATA_MODE_SHMGET)
        # self.prefill_decoder_top_o_0bindings['past_key0'] = prefill_decoder_top_o_binding_k
        # prefill_decoder_top_o_binding_v = self.graph_engine.get_rpp_fw_out_one(graph_decoders_topx,
        #                                                                     'past_value0',
        #                                                                     1*8*128*128*2*mul_ratio,
        #                                                                     self.IO_DATA_MODE_SHMGET)
        # self.prefill_decoder_top_o_0bindings['past_value0'] = prefill_decoder_top_o_binding_v

        # prefill mid in
        self.prefill_decoder_mid_i_0bindings = OrderedDict()
        graph_decoders_mid0: str = f'{self.prefill_mid_name}0_8192'
        prefill_decoder_mid_i_binding_2 = self.graph_engine.get_rpp_fw_in_one(graph_decoders_mid0,
                                                                                '/self_attn/Slice_4_output_0',
                                                                                1*1*128*self.input_size*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_mid_i_0bindings['/self_attn/Slice_4_output_0'] = prefill_decoder_mid_i_binding_2
        
        # prefill top out
        self.prefill_decoder_top_o_31bindings = self._alloc_graph_io_params(f'{self.prefill_bot_name}31', 2, self.IO_DATA_MODE_NULL)
        
        # prefill rms
        self.prefill_rms_i_bindings, self.prefill_rms_o_bindings = self._alloc_graph_io_params(self.prefill_rms_norm_name, 3)

        # prefill lm_head
        self.prefill_lm_heads_o_bindings_list = OrderedDict()
        self.prefill_lm_heads_o_bindings_list['3'] = self.graph_engine.get_rpp_fw_out_one(f'{self.prefill_lm_head_name}{0}',
                                                                                        '3',
                                                                                        1*128256*4,
                                                                                        self.IO_DATA_MODE_SHMGET)
        
        self.prefill_k_i_bindings, self.prefill_k_o_bindings = self._alloc_graph_io_params(f'{self.prefill_k_quant_name}0', 3)
        self.prefill_v_i_bindings, self.prefill_v_o_bindings = self._alloc_graph_io_params(f'{self.prefill_v_quant_name}0', 3)
        
        # embed tokens
        self.prefill_embed_i_bindings = OrderedDict()
        prefill_embed_i_binding_0 = self.graph_engine.get_rpp_fw_in_one(self.prefill_embed_name,
                                                                        'embedding_input',
                                                                        4096*2,
                                                                        self.IO_DATA_MODE_SHMGET)
        self.prefill_embed_i_bindings['embedding_input'] = prefill_embed_i_binding_0
        
    def _build_cpu_models(self):
        r"""
        Build the other models which is running in cpu, include prefill rotary_emb onnx  and decode rotary_emb onnx
        """
        self.prefill_rotary_emb = {
            '1024': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', 'loop1_rotary_emb_sim.onnx')),
            '4096': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', 'loop1_rotary_emb_4096_sim.onnx')),
            '8192': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', 'loop1_rotary_emb_8192_sim.onnx'))
        }
        self.decode_rotary_emb = onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', 'loop2_rotary_emb_sim.onnx'))
        # try:
        #     self.embed_tokens = torch.jit.load(os.path.join(self.rpp_dir, 'model_files', 'embed_tokens.pt'), map_location='cpu').eval()
        # except:
        #     self.embed_tokens = torch.load(os.path.join(self.rpp_dir, 'model_files', 'embed_tokens.pt'), map_location='cpu').eval()
        
    def _update_prefill_top_param(self, loop, index):
        r"""
        Load and exec top dynamic graph for prefill stage
        
        Args:
            loop (`int`):
                The index of current top
            index (`int`):
                The index of current top dynamic graph
        """
        graph_top_param_x: str = f'{self.prefill_top_param_name}/top{loop}/graph_param_{index}'
        if index == 0:
            self._graph_load_sharedID(graph_top_param_x, self.global_share_id[self.prefill_top_param_name])
        self._graph_execute_sync(graph_top_param_x)
    
    def _update_prefill_mid_param(self, loop, index): 
        r"""
        Load and exec mid dynamic graph for prefill stage
        
        Args:
            loop (`int`):
                The index of current mid
            index (`int`):
                The index of current mid dynamic graph
        """     
        if self.prompt_len < 1024:
            mid_name = 'mid0_1024'
        elif self.prompt_len < 4096:
            mid_name = 'mid0_4096'
        else:
            mid_name = 'mid0_8192'
        graph_mid_param_x: str = f'{self.prefill_mid_param_name}/{mid_name}/graph_param_{index}'
        if index == 0:
            self._graph_load_sharedID(graph_mid_param_x, self.global_share_id[self.prefill_mid_param_name])
        self._graph_execute_sync(graph_mid_param_x)
        
    def _update_prefill_bot_param(self, loop, index):
        r"""
        Load and exec bot dynamic graph for prefill stage
        
        Args:
            loop (`int`):
                The index of current bot
            index (`int`):
                The index of current bot dynamic graph
        """     
        graph_bot_param_x: str = f'{self.prefill_bot_param_name}/bot{loop}/graph_param_{index}'
        if index == 0:
            self._graph_load_sharedID(graph_bot_param_x, self.global_share_id[self.prefill_bot_param_name])
        self._graph_execute_sync(graph_bot_param_x)
    
    def _run_prefill_top(self, loop, idx, range):
        r"""
        Load and exec bot static graph for prefill stage
        
        Args:
            loop (`int`):
                The index of current bot
            idx (`int`):
                The index of current bot dynamic graph
            range (`int`):
                The number of current bot dynamic graph
        """   
        graph_decoders_topx: str = f'{self.prefill_top_name}{loop}'
        self._graph_exec(graph_decoders_topx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        if idx < range - 1:
            graph_top_param_x: str = f'{self.prefill_top_param_name}/top{loop}/graph_param_{idx + 1}'
            self._graph_load_sharedID(graph_top_param_x, self.global_share_id[self.prefill_top_param_name])
        self._graph_poll(graph_decoders_topx)
             
    def _run_prefill_mid(self, loop, idx, range):
        r"""
        Load and exec mid static graph for prefill stage
        
        Args:
            loop (`int`):
                The index of current mid
            idx (`int`):
                The index of current mid dynamic graph
            range (`int`):
                The number of current mid dynamic graph
        """   
        if self.prompt_len < 1024:
            graph_decoders_midx: str = f'{self.prefill_mid_name}0_1024'
            mid_param_name = 'mid0_1024'
        elif self.prompt_len < 4096:
            graph_decoders_midx: str = f'{self.prefill_mid_name}0_4096'
            mid_param_name = 'mid0_4096'
        else:
            mid_param_name = 'mid0_8192'
            graph_decoders_midx: str = f'{self.prefill_mid_name}0_8192'
        self._graph_exec(graph_decoders_midx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        if idx < range - 1:
            graph_mid_param_x: str = f'{self.prefill_mid_param_name}/{mid_param_name}/graph_param_{idx + 1}'
            self._graph_load_sharedID(graph_mid_param_x, self.global_share_id[self.prefill_mid_param_name])
        self._graph_poll(graph_decoders_midx)
    
    def _run_prefill_bot(self, loop, idx, range):
        r"""
        Load and exec bot static graph for prefill stage
        
        Args:
            loop (`int`): 
                The index of current bot
            idx (`int`):
                The index of current bot dynamic graph
            range (`int`):
                The number of current bot dynamic graph
        """   
        graph_decoders_botx: str = f'{self.prefill_bot_name}{loop}'
        self._graph_exec(graph_decoders_botx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        if idx < range - 1:
            graph_bot_param_x: str = f'{self.prefill_bot_param_name}/bot{loop}/graph_param_{idx + 1}'
            self._graph_load_sharedID(graph_bot_param_x, self.global_share_id[self.prefill_bot_param_name])
        self._graph_poll(graph_decoders_botx)
        
    def _run_prefill_kv_reformat(self):
        r"""
        Reformat kv from top to mid
        """   
        if self.prompt_len < 1024:
            k_name = f'{self.prefill_k_reformat_name}_1024'
            v_name = f'{self.prefill_v_reformat_name}_1024'
        elif self.prompt_len < 4096:
            k_name = f'{self.prefill_k_reformat_name}_4096'
            v_name = f'{self.prefill_v_reformat_name}_4096'
        else:
            k_name = f'{self.prefill_k_reformat_name}_8192'
            v_name = f'{self.prefill_v_reformat_name}_8192'
        repeat_times = 8
        k_params = f'{k_name}/graph_param_0'
        self._graph_load_sharedID(k_params, self.global_share_id[self.prefill_k_reformat_name])
        for i in range(repeat_times):
            self._graph_execute_sync(k_params)
            self._graph_exec(k_name, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if i < repeat_times - 1:
                k_params = f'{k_name}/graph_param_{i + 1}'
                self._graph_load_sharedID(k_params, self.global_share_id[self.prefill_k_reformat_name])
            self._graph_poll(k_name)
        
        v_params = f'{v_name}/graph_param_0'
        self._graph_load_sharedID(v_params, self.global_share_id[self.prefill_v_reformat_name])
        for j in range(repeat_times):
            self._graph_execute_sync(v_params)
            self._graph_exec(v_name, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if j < repeat_times - 1:
                v_params = f'{v_name}/graph_param_{j + 1}'
                self._graph_load_sharedID(v_params, self.global_share_id[self.prefill_v_reformat_name])
            self._graph_poll(v_name)
    
    def _run_prefill_decoders(self, loop):
        r"""
        Do prefill inference on rpp
        
        Args:
            loop(`int`):
                The current number of cycles
        """   
        next_len = self.question_len + 1
        prefill_range = next_len // 128 if next_len % 128 == 0 else next_len // 128 + 1
        # need set warmup in prefill
        if loop == 0:
            self._warmup(prefill_range)
               
        for idx in range(prefill_range):
            self._update_prefill_top_param(loop, idx)
            self._run_prefill_top(loop, idx, prefill_range)
        
        self._run_prefill_kv_reformat()
        k_scales, v_scales = self._run_prefill_kv_quant(loop, self.prompt_len + 1)
        
        mid_range = prefill_range * 4 if self.prompt_len >= 4096 else prefill_range
        for idx in range(mid_range):
            self._update_prefill_mid_param(0, idx)
            self._run_prefill_mid(0, idx, mid_range)

        for idx in range(prefill_range):
            self._update_prefill_bot_param(loop, idx)
            self._run_prefill_bot(loop, idx, prefill_range)
        return True, k_scales, v_scales

    def _init_kv_quant_data(self, name = 'key'):
        r"""
        Copy kv data from top
        
        Args:
            name(`str`): 
                key or value 
        """ 
        N = int(math.ceil((self.prompt_len + 1) / 256) * 256)
        top_out_bindings_name = 'past_key0' if name == 'key' else 'past_value0'
        top_out  = self.graph_engine.save_int16_to_int16_numpy(self.prefill_decoder_top_o_0bindings[top_out_bindings_name].tensor_haddr,
                                                               self.prefill_decoder_top_o_0bindings[top_out_bindings_name].tensor_daddr,
                                                               N*8*128*2,
                                                               self.prefill_decoder_top_o_0bindings[top_out_bindings_name].mapped)
        
        return top_out
    
    def _run_prefill_kv_quant(self, kv_arrays: Tuple[Tuple[np.ndarray]], token_length: int):
        r"""
        Old interface, not use, do kv quantization
        
        Args:
            kv_arrays (`Tuple`): 
                Kv information
            token_length (`int`): 
                Current tokens length
        """ 
        # kv_arrays = kv_arrays.transpose(0, 1, 3, 2, 4, 5).reshape(32, 2, 8, -1, 128)
        # kv_arrays[:, :, :, token_length:, :] = 0.
        # kv_arrays = kv_arrays.reshape(32, 2, 8, -1, 128, 128).transpose(0, 1, 3, 2, 4, 5)
        kv_scales = []
        if token_length % 256 == 0:
            repeat_num = token_length // 256
        else:
            repeat_num = (token_length // 256) + 1
        self._warmup(0, address=[0x6000814, 0x6001814])
        self._warmup(0, address=[0x6000818, 0x6001818])
        for idx in range(32):
            tmp_quant_k: str = f'{self.prefill_k_quant_name}{idx}'
            tmp_quant_v: str = f'{self.prefill_v_quant_name}{idx}'
            # k_data = kv_arrays[idx][0]
            # v_data = kv_arrays[idx][1]
            # self.graph_engine.copy_data_to_device(k_data.ravel(), k_data.nbytes, self.prefill_k_i_bindings['input'].tensor_daddr)
            # self.graph_engine.copy_data_to_device(v_data.ravel(), v_data.nbytes, self.prefill_v_i_bindings['input'].tensor_daddr)
            
            k_scales = []
            v_scales = []
            graph_k_param_x = f'{tmp_quant_k}/graph_param_0'
            self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.prefill_k_quant_name])
            for i in range(repeat_num):
                self._graph_execute_sync(graph_k_param_x)
                self._graph_exec(tmp_quant_k, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
                if i < repeat_num - 1:
                    graph_k_param_x = f'{tmp_quant_k}/graph_param_{i + 1}'
                    self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.prefill_k_quant_name])
                self._graph_poll(tmp_quant_k)
                tmp_scales = self.graph_engine.save_int16_to_int16_numpy(self.prefill_k_o_bindings['output0'])
                k_scales.append(tmp_scales)
            
            graph_v_param_x = f'{tmp_quant_v}/graph_param_0'
            self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.prefill_v_quant_name])
            for j in range(repeat_num):
                self._graph_execute_sync(graph_v_param_x)
                self._graph_exec(tmp_quant_v, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
                if j < repeat_num - 1:
                    graph_v_param_x = f'{tmp_quant_v}/graph_param_{j + 1}'
                    self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.prefill_v_quant_name])
                self._graph_poll(tmp_quant_v)
                tmp_scales = self.graph_engine.save_int16_to_int16_numpy(self.prefill_v_o_bindings['output0'])
                v_scales.append(tmp_scales)

            k_scales = np.concatenate(k_scales, axis=0)
            v_scales = np.concatenate(v_scales, axis=0)
            pad_factor = self.input_size - k_scales.shape[0]
            k_scales = np.pad(k_scales, ((0, pad_factor)), 'constant', constant_values=0)
            v_scales = np.pad(v_scales, ((0, pad_factor)), 'constant', constant_values=0)

            kv_scales.append(k_scales)
            kv_scales.append(v_scales)
        scales = np.concatenate(kv_scales)
        return scales
    
    def _run_prefill_kv_quant(self, idx: int, token_length: int):
        r"""
        Do kv quantization
        
        Args:
            idx(`int`): 
                Current prefill index
            token_length(`int`): 
                Current tokens length
        """ 
        if token_length % 256 == 0:
            repeat_num = token_length // 256
        else:
            repeat_num = (token_length // 256) + 1
        self._warmup(0, address=[0x6000814, 0x6001814])
        self._warmup(0, address=[0x6000818, 0x6001818])

        tmp_quant_k: str = f'{self.prefill_k_quant_name}{idx}'
        tmp_quant_v: str = f'{self.prefill_v_quant_name}{idx}'
        k_scales = []
        v_scales = []
        graph_k_param_x = f'{tmp_quant_k}/graph_param_0'
        self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.prefill_k_quant_name])
        for i in range(repeat_num):
            self._graph_execute_sync(graph_k_param_x)
            self._graph_exec(tmp_quant_k, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if i < repeat_num - 1:
                graph_k_param_x = f'{tmp_quant_k}/graph_param_{i + 1}'
                self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.prefill_k_quant_name])
            self._graph_poll(tmp_quant_k)
            tmpk_scales = self.graph_engine.save_int16_to_int16_numpy(self.prefill_k_o_bindings['output0'])
            k_scales.append(tmpk_scales)
        
        graph_v_param_x = f'{tmp_quant_v}/graph_param_0'
        self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.prefill_v_quant_name])
        for j in range(repeat_num):
            self._graph_execute_sync(graph_v_param_x)
            self._graph_exec(tmp_quant_v, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if j < repeat_num - 1:
                graph_v_param_x = f'{tmp_quant_v}/graph_param_{j + 1}'
                self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.prefill_v_quant_name])
            self._graph_poll(tmp_quant_v)
            tmpv_scales = self.graph_engine.save_int16_to_int16_numpy(self.prefill_v_o_bindings['output0'])
            v_scales.append(tmpv_scales)
            
        k_scales = np.concatenate(k_scales, axis=0)
        v_scales = np.concatenate(v_scales, axis=0)
        pad_factor = self.target_len - k_scales.shape[0]
        k_scales = np.pad(k_scales, ((0, pad_factor)), 'constant', constant_values=0)
        v_scales = np.pad(v_scales, ((0, pad_factor)), 'constant', constant_values=0)

        return k_scales, v_scales
    
    def _run_embed_tokens(self, token_id: int):
        r"""
        Get embed data from device
        
        Args:
            token_id(`int`): 
                Current token number
        """ 
        offset = 4096 * 2
        daddr = self.prefill_embed_i_bindings['embedding_input'].tensor_daddr + offset * token_id
        haddr = self.prefill_embed_i_bindings['embedding_input'].tensor_haddr
        share = self.prefill_embed_i_bindings['embedding_input'].mapped
        hidden_states = self.graph_engine.save_bf16_to_numpy(haddr, daddr, offset, share)
        return hidden_states
    
    def _run_prefill_nor_head(self):
        r"""
        Run rms norm and lm head on rpp
        """ 
        # self.prefill_host_buffer_list.clear()
        # if self.prompt_len % 256 == 0:
        #     rms_range = self.prompt_len // 256
        # else:
        #     rms_range = self.prompt_len // 256 + 1
        # # update params of rms, only run last one
        # graph_rms_param_x: str = f'{self.prefill_rms_param_name}/graph_param_{rms_range - 1}'
        # self._graph_load_sharedID(graph_rms_param_x, self.global_share_id[self.prefill_rms_param_name])
        # self._graph_execute_sync(graph_rms_param_x)
        # # exec rms
        # self._graph_execute_sync(self.prefill_rms_norm_name)

        # for j in range(4):
        #     prefill_lm_head_x: str = f'{self.prefill_lm_head_name}{j}'
        #     self._graph_execute_sync(prefill_lm_head_x)
        
        # lm_out = self.graph_engine.save_fp32_to_numpy(self.prefill_lm_heads_o_bindings_list['3'])
        # self.prefill_host_buffer_list.append(lm_out)
        
        # void copy_data_to_device(py::buffer& buf, size_t data_size, uintptr_t daddr, size_t ddr_offset)
        # void save_data_to_host(py::buffer& buf, size_t data_size, uintptr_t daddr, size_t ddr_offset)
        
        self.prefill_host_buffer_list.clear()
        rms_norm_offset = (self.prompt_len-1) * 4096 * 4
        hidden_out = np.zeros((1,1,4096), np.float32)
        self.graph_engine.save_data_to_host(hidden_out, hidden_out.nbytes, self.prefill_decoder_top_o_31bindings['hidden_out'].tensor_daddr + rms_norm_offset)
        self.graph_engine.copy_data_to_device(hidden_out, hidden_out.nbytes, self.prefill_rms_i_bindings['input'].tensor_daddr)
        self._graph_execute_sync(self.prefill_rms_norm_name)
        for j in range(4):
            prefill_lm_head_x: str = f'{self.prefill_lm_head_name}{j}'
            self._graph_execute_sync(prefill_lm_head_x)
        lm_out = self.graph_engine.save_fp32_to_numpy(self.prefill_lm_heads_o_bindings_list['3'])
        self.prefill_host_buffer_list.append(lm_out)
        return True
    
    def _print_prefill_progress(self):
        r"""
        Print prefill progress bar
        """
        finsh = "▓" * self.cur_count
        self.cur_count = self.cur_count + 1
        need_do = "-" * (self.all_count - self.cur_count)
        progress = (self.cur_count / self.all_count) * 100
        dur = time.perf_counter() - self.prefill_start_time
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(progress, finsh, need_do, dur), end="")
        if self.cur_count == self.all_count:
            print("")

    def _copy_prefill_init_data(self, prefill_input):
        r"""
        Copy the init data to device for prefill stage
        
        Args:
            prefill_input(`List`): 
                The init data
        """
        self.graph_engine.copy_fp32_from_numpy(self.prefill_decoder_top_i_0bindings["hidden_in"], prefill_input[0])
        self.graph_engine.copy_fp32_from_numpy(self.prefill_decoder_top_i_0bindings["/self_attn/Unsqueeze_output_0"], prefill_input[1])
        self.graph_engine.copy_fp32_from_numpy(self.prefill_decoder_top_i_0bindings["/self_attn/Unsqueeze_1_output_0"], prefill_input[2])
        self.graph_engine.copy_fp32_from_numpy(self.prefill_decoder_mid_i_0bindings['/self_attn/Slice_4_output_0'], prefill_input[3])
        
    def _run_prefill_inference(self, input_ids, original_input_ids, attention_mask, position_ids, ppl_flag=False):
        r"""
        Run prefill inference on rpp, include decode, rms, lmhead and so on
        
        Args:
            input_ids: 
                Input ids
            original_input_ids: 
                Original input ids
            attention_mask: 
                Attention mask
            position_ids: 
                Position ids
        """
        self.prompt_len = len(original_input_ids)
        hidden_states = []
        current_states = None
        cached_states = None
        if self.low_power:
            self._graph_set_power_mode(0)
        for i in range(input_ids.shape[-1]):
            if i <= len(original_input_ids):
                token_id = input_ids[0, i].tolist()
                current_states = self._run_embed_tokens(token_id)
                if i == len(original_input_ids):
                    cached_states = current_states.copy()
            else:
                current_states = cached_states
            hidden_states.append(current_states)
        hidden_states = np.stack(hidden_states, axis=0)
        # hidden_states = self.embed_tokens(input_ids).detach().numpy()
        attention_mask = attention_mask.detach().numpy()
        position_ids = position_ids.detach().numpy()
        
        if len(original_input_ids) < 1024:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['1024'].run(['cosine', 'sine'], {'position_ids': position_ids})
        elif len(original_input_ids) < 4096:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['4096'].run(['cosine', 'sine'], {'position_ids': position_ids})
        else:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['8192'].run(['cosine', 'sine'], {'position_ids': position_ids})
        
        # run prefill decoders
        prefill_input = []
        prefill_input.append(hidden_states.ravel())
        prefill_input.append(gather_out_0.ravel())
        prefill_input.append(gather_out_1.ravel())
        prefill_input.append(attention_mask.ravel())
        self._copy_prefill_init_data(prefill_input)
        kv_scales = []
        self._warmup(len(original_input_ids) + 1, address=[0x6000810, 0x6001810])
        for i in range(32):
            if self.stop_infer:
                return None
            self._print_prefill_progress()
            output = self._run_prefill_decoders(i)
            if not output[0]:
                return None
            kv_scales.append(output[1])
            kv_scales.append(output[2])
        self._run_prefill_nor_head()
        self._print_prefill_progress()
        self._warmup(int(math.ceil((len(original_input_ids) + 1) / 256)), address=[0x6000804, 0x6001804])
        self._warmup(32 * 2, address=[0x6000808, 0x6001808])
        self._warmup(0, address=[0x600080C, 0x600180C])
        # push the quant_scales
        self._run_kv_init(np.concatenate(kv_scales))
        if self.low_power:
            self._graph_set_power_mode(1)

        lm_head_outs = self.prefill_host_buffer_list[0].reshape(1, -1)
        token = self._get_next_token(original_input_ids, lm_head_outs)
        self._print_prefill_progress()
        if ppl_flag:
            return lm_head_outs
        elif self.write_file:
            print("inference......, the result will write to the prompt_result.txt file.")
            return token
        else:
            try:
                self.cached_token.append(token.tolist()[0])
                out_str = self.tokenizer.decode(self.cached_token, skip_special_tokens=True)
                if '�' not in out_str:
                    if self.print_tokens:
                        print(out_str, end='', flush=True)
                    self.str_tokens = self.str_tokens + out_str
                    self.cached_token.clear()
            except (UnicodeDecodeError,AttributeError) as e:
                print(token)
                print('except:', e)
                print(self.tokenizer.decoder[token.tolist()[0]])
        return token
        
    def _run_kv_init(self, in_data):
        r"""
        Push the quant scales data to device
        
        Args:
            in_data: 
                Quant scales data
        """
        self.graph_engine.copy_int16_from_int16_numpy(self.decode_kv_i_bindings['input'], in_data)
        self._graph_execute_sync(self.decode_kv_name)
        
    def _update_decode_dynamic_param(self, loop, last_flag = False):
        r"""
        Update decode dynamic param for each round of decoding
        
        Args:
            loop (`int`): 
                current tokens number
        """
        if loop < self.target_len:
            graph_param_x: str = f'{self.decode_param_name}{loop}'
            self._graph_execute_sync(graph_param_x)
        
    def _init_decode_dynamic_param(self, loop):
        r"""
        Init decode dynamic param
        
        Args:
            loop (`int`): 
                current tokens number
        """
        graph_param_x: str = f'{self.decode_param_name}{(loop // 256)*256}'
        self._graph_load_sharedID(graph_param_x, self.global_share_id[self.decode_param_name])
        self._graph_execute_sync(graph_param_x)
        graph_param_x: str = f'{self.decode_param_name}{loop}'
        self._graph_load_sharedID(graph_param_x, self.global_share_id[self.decode_param_name])

    def _copy_decode_init_data(self):
        r"""
        Copy the data top device for each round of decoding 
        """
        self.graph_engine.copy_fp32_from_numpy(self.decode_decoder_i_bindings["hidden_in"], self.hidden_states_bindings)
        self.graph_engine.copy_fp32_from_numpy(self.decode_decoder_i_bindings["attention_mask"], self.attention_mask_bindings)
        self.graph_engine.copy_fp32_from_numpy(self.decode_decoder_i_bindings['/self_attn/Unsqueeze_output_0'], self.gather_out_0)
        self.graph_engine.copy_fp32_from_numpy(self.decode_decoder_i_bindings['/self_attn/Unsqueeze_1_output_0'], self.gather_out_1)
    
    def _run_decode_inference(self, loop):
        r"""
        Do decode inference on rpp
        
        Args:
            loop (`int`): 
                Current decode loop
        """
        self._copy_decode_init_data()
        for i in range(HEAD_NUM):
            decoders_x: str = f'{self.decode_decodes_name}{i}'
            if i == 0:
                self._graph_exec(decoders_x, UPDATE_HEAD)
            else:
                self._graph_exec(decoders_x, UPDATE_TAIL)
        self._run_decode_nor_head(loop)
    
    def _run_decode_nor_head(self, loop):
        r"""
        Run rms nor and lm head for decode on rpp
        
        Args:
            loop (`int`): 
                Current decode loop
        """
        graph_param_x: str = f'{self.decode_param_name}{loop + 1}'
        self._graph_exec(self.decode_rms_norm_name, UPDATE_TAIL + UPDATE_INTE)
        if loop + 1 < self.target_len:
            self._graph_load_sharedID(graph_param_x, self.global_share_id[self.decode_param_name])
        self._graph_poll(self.decode_rms_norm_name)
        
        self.decode_host_buffer_list.clear()
        for idx in range(4):
            graph_lm_head_x: str = f'{self.decode_lm_head_name}{idx}'
            self._graph_exec(graph_lm_head_x, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if idx > 0:
                output = self.graph_engine.save_fp32_to_numpy(self.decode_lm_heads_o_bindings_list[idx-1]['3'])
                self.decode_host_buffer_list.append(output)
            self._graph_poll(graph_lm_head_x)
            if idx == 3:
                output = self.graph_engine.save_fp32_to_numpy(self.decode_lm_heads_o_bindings_list[idx]['3'])
                self.decode_host_buffer_list.append(output)
        return True

    def _run_prefill_kv_reformat_cpu(self, tensors: List[np.ndarray], repeats: int, magic_num: int=8) -> List[np.ndarray]:
        r"""
        Post-process the Key or Value tensor list([key, value]) from the RPP prefill top model output in shape [1048576,], re-order
        the tensor to chw32 format.
            i.e.
                repeats = 8
                Key:
                    [1048576,] -> [repeats, 8, 128(0), 128(1)] -> [8, repeats, 128(0), 128(1)] -> [8, 128(1), repeats*128(0)] -> 
                    [8, 128(1), -1, 32] -> [8, -1, 128(1), 32]
                Value:
                    [1048576,] -> [repeats, 8, 128(0), 128(1)] -> [8, repeats, 128(0), 128(1)] -> [8, repeats*128(0), 128(1)] -> 
                    [8, repeats*128(0), -1, 32] -> [8, -1, repeats*128(0), 32]
                    
        Args:
            tenosr (`List[np.ndarray]`): 
                The Key or Value tensor
            repeats (`int`):
                The prefill top model repeat times
            magic_num (`int`):
                The magic number for reshape the np.ndarray
        Returns: 
            tensor (`List[np.ndarray]`):
                The post-processed Key or Value tensor
        """
        # ------------------------ Key -------------------------------------
        # step 1: reshape the Key tensor to [magic_num, 4, 128(0), 128(1)]
        # step 2: transpose the tensor to [4, 128(1), magic_num, 128(0)]
        # step 3: reshape the tensor to [4, 128(1), magic_num*128(0)]
        # step 4: reshape the tensor to [4, 128(1), -1, 32]
        # step 5: transpose the tensor to [4, -1, 128(1), 32]
        key = tensors[0].reshape(-1, 8, 128, 128).transpose(1, 3, 0, 2).reshape(8, 128, -1)[:, :, :repeats*128].reshape(8, 128, -1, 32).transpose(0, 2, 1, 3)
        # tensors[0] = tensors[0].reshape(magic_num, 8, 128, 128).transpose(1, 3, 0, 2).reshape(8, 128, -1)[:, :, :repeats*128].reshape(8, 128, -1, 32).transpose(0, 2, 1, 3)
        pad_factor = ((0, 8-key.shape[0]), (0, self.input_size//32-key.shape[1]), (0, 128-key.shape[2]), (0, 32-key.shape[3]))
        key = np.pad(key, pad_factor, mode='constant', constant_values=0)

        # ------------------------ Value -------------------------------------
        # step 1: reshape the Value tensor to [magic_num, 4, 128(0), 128(1)]
        # step 2: transpose the tensor to [4, magic_num, 128(0), 128(1)]
        # step 3: reshape the tensor to [4, magic_num*128(0), 128(1)]
        # step 4: reshape the tensor to [4, magic_num*128(0), -1, 32]
        # step 5: transpose the tensor to [4, -1, magic_num*128(0), 32]
        value = tensors[1].reshape(-1, 8, 128, 128).transpose(1, 0, 2, 3)
        dims = value.shape[1]
        value = value.reshape(8, -1, 128).reshape(8, dims*128, -1, 32)[:, :repeats*128, ...].transpose(0, 2, 1, 3)
        # tensors[1] = tensors[1].reshape(magic_num, 8, 128, 128).transpose(1, 0, 2, 3).reshape(8, -1, 128).reshape(8, magic_num*128, -1, 32)[:, :repeats*128, ...].transpose(0, 2, 1, 3)
        pad_factor = ((0, 8-value.shape[0]), (0, 4-value.shape[1]), (0, self.input_size-value.shape[2]), (0, 32-value.shape[3]))
        value = np.pad(value, pad_factor, mode='constant', constant_values=0)

        return [key, value]
    
    def _run_prefill_kv_quant_cpu(self, kv_arrays: Tuple[Tuple[np.ndarray]], token_length: int) -> Tuple[np.ndarray]:
        r"""
        Quantize the Key & Value tensor for decode decoders inference.

        Args:
            kv_arrays (`Tuple[Tuple[np.ndarray]]`): 
                The Key & Value tensor in shape [32, 2]
            token_length (`int`): 
                The real token length of the prefill inference results
        Returns 
            quant_past_key, quant_past_value, scales (`Tuple[np.ndarray]`):
            quant_past_key (`np.ndarray`): 
                The quantized Key tensor in shape (32, 8, 128, 2048)
            quant_past_value (`np.ndarray`): 
                The quantized Value tensor in shape (32, 8, 2048, 128)
            scales (`np.ndarray`): 
                The quantized scales for Key & Value tensor in shape (131072,)
        """
        # K: [N * 8 * 256 * 128], V: [N * 8 * 256 * 128]
        # 32 is 32 decoders; 2 is for kv, 1 is the batch num
        N = int(math.ceil(token_length / 256))
        kv_data = np.array(kv_arrays)[:, :, :, :, :token_length, :]  # [32, 2, 1, 8, 27, 128]
        pad_factor = int(N*256) - kv_data.shape[4]
        # kv_arrays chanege  (32, 2, 1, 8, token_length, 128) -> (32, 2, 1, 8, token_length:pad_zero, 128)
        # [32, 2, 1, 8, N*256, 128]
        kv_data = np.pad(kv_data, ((0,0), (0,0), (0,0), (0,0), (0, pad_factor), (0,0)), 'constant', constant_values=0)
        kv_data = kv_data.reshape((32, 2, 1, 8, -1, 256, 128)) # [32, 2, 1, 8, N, 256, 128]
        kv_scales = []

        kv_data = kv_data.transpose(0, 1, 2, 4, 5, 3, 6).reshape(32, 2, 1, N, 256, -1)
        kv_max = np.max(np.abs(kv_data), axis=-1)
        kv_mask = kv_max == 0.0
        kv_scales = 127.0 / kv_max
        kv_scales[kv_mask] = 0.0

        kv_quants = np.clip((kv_data * kv_scales[..., np.newaxis]).round(), -127.0, 127.0).astype(np.int8)

        kv_scales = 1.0 / kv_scales
        kv_scales[kv_mask] = 0.0
        kv_scales = kv_scales.reshape(32, 2, 1, -1)
        pad_factor = self.input_size - kv_scales.shape[-1]
        kv_scales = np.pad(kv_scales, ((0,0), (0,0), (0,0), (0, pad_factor)), 'constant', constant_values=0)

        k_data = kv_quants[:, 0, 0, ...]
        v_data = kv_quants[:, 1, 0, ...]

        # [HEAD_NUM, N*256, 8*128] -> [HEAD_NUM, N, 8, 128, 256]
        k_data = k_data.reshape(32, N, 256, 8, 128).transpose(0, 1, 3, 4, 2)
        # [HEAD_NUM, N*256, 8*128] -> [HEAD_NUM, N, 8, 256, 128]
        v_data = v_data.reshape(32, N, 256, 8, 128).transpose(0, 1, 3, 2, 4)

        return k_data, v_data, kv_scales.ravel()
 
    def _make_causal_mask(self,
                          input_ids_shape: torch.Size, 
                          dtype: torch.dtype, 
                          device: torch.device,
                          past_key_values_length: int = 0) -> torch.Tensor:
        r"""
        Make causal mask used for bi-directional self-attention.
        
        Args:
            input_ids_shape (`torch.Size`): 
                The shape information for processing the mask tensor
            dtype (`torch.dtype`): 
                The target dataType of the mask tensor
            device (`torch.device`): 
                The target deivce location ('cpu' or 'cuda') of the mask tensor
            past_key_values_length (`int`): 
                The length of the Key & Value tensors
        Returns: 
            mask (`torch.Tensor`): 
                The processed attention_mask tensor
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self,
                     mask: torch.Tensor, 
                     dtype: torch.dtype, 
                     device: torch.device,
                     tgt_len: int) -> torch.Tensor:
        r"""
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

        Args:
            mask (`torch.Tensor`):
                The attention_mask tensor is waiting to expand
            dtype (`torch.dtype`):
                The dataType of the target tensor
            device (`torch.device`):
                The device location ('cpu' or 'cuda') of the target tensor
            tgt_len (`int`):
                The target length of expanding operation

        Returns:
            inverted_mask (`torch.Tensor`):
                The expanded attention_mask tensor
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min).to(device)
   
    def _get_attetion_mask(self, 
                          attention_mask: torch.Tensor, 
                          input_shape: Tuple[int],
                          target_dtype: torch.dtype,
                          target_device: torch.device, 
                          past_key_values_length: int) -> torch.Tensor:
        r"""
        Process the attention_mask tenosr for decoders.

        Args:
            attention_mask (`torch.Tensor`):
                Original attention mask tensor waiting for processing
            input_shape (`Tuple[int]`):
                The shape used to process the attention mask tensor
            target_dtype (`torch.dtype`):
                The target dataType of the processed attention_mask tensor
            target_device (`torch.device`):
                The target device location ('cpu' or 'cuda') of the processed attention_mask tensor
            past_key_values_length (`int`):
                The length of the Key & Value tensors

        Returns:
            combined_attention_mask (`torch.Tensor`):
                The processd attention_mask tensor
        """
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                target_dtype,
                target_device,
                past_key_values_length)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, 
                                                   target_dtype,
                                                   target_device,
                                                   tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def _apply_chat_prompt(self, prompt: Union[str, List], tools: List=None):
        r"""
        Tokenizer prompt information

        Args:
            prompt (`Union[str, List]`):
                The input information from user
            tools (`List`):
                Tools function
        Returns:
            input_ids_padding (`Tensor`):
                The padding input ids
            input_ids (`Tensor`):
                Origin input ids
            attention_mask (`Tensor`):
                The attention mask
            position_ids (`Tensor`)
                The position ids
        """
        if isinstance(prompt, list):
            conversation = prompt
        else:
            conversation = [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}]
        if self.prefix:
            for i in range(len(conversation)):
                tmp_conversation = conversation[i:]
                inputs = self.tokenizer.apply_chat_template(tmp_conversation,
                                                            tools=tools,
                                                            add_generation_prompt=True,
                                                            tokenize=True,
                                                            return_dict=True,
                                                            return_tensors="pt",
                                                            truncation=True,
                                                            max_length=self.input_size - 2,
                                                            padding=False)
                if inputs.input_ids.shape[-1] < self.input_size - 1:
                    break
                if i == len(conversation) - 1:
                    break
            input_ids = inputs.input_ids
            tmp_attention_mask = inputs.attention_mask
        else:
            ##no need to add prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt)])
            tmp_attention_mask = torch.full((1, input_ids.shape[-1]), 1, dtype=torch.int)

        if input_ids.shape[-1] >= 4096:
            input_size = 8192
        elif input_ids.shape[-1] >= 1024:
            input_size = 4096
        else:
            input_size = 1024
        
        input_len = input_ids.shape[-1]
        right_padding = input_size - input_len
        
        if right_padding - 1 <= 0:
            print(f'The prompt token max length need < {self.input_size - 1} your prompt token length is {input_len}')
            print(f'Please input your prompt again')
            return None, None, None, None
        
        # padding the input_ids, 128009 is the magic number of Llama3, which is the total embedding space of Llama3 model
        fixed_tensor = torch.full((1, right_padding), 128009, dtype=torch.int)

        # concat the tensors & get the real input_ids
        input_ids_padding = torch.cat([input_ids, fixed_tensor], 1)

        # get the attention_mask
        # attention_mask_left = torch.tensor(np.ones((1, input_len), dtype=np.float32))
        attention_mask_left = tmp_attention_mask
        attention_mask_right = torch.tensor(np.zeros((1, right_padding), dtype=np.float32))
        attention_mask_com = torch.cat([attention_mask_left, attention_mask_right], 1)
        attention_mask = self._get_attetion_mask(attention_mask_com, 
                                                 (1, input_size), 
                                                 torch.float32, 
                                                 torch.device('cpu'), 
                                                 0)
        
        position_ids = torch.tensor(np.arange(input_size, dtype=np.int64).reshape((1, input_size)))
        input_ids = input_ids.squeeze()
        print(f"original input_id size is {input_ids.shape[-1]}")

        return input_ids_padding, input_ids, attention_mask, position_ids

    def rpp_inference(self, **kwargs):
        r"""
        Inference the prompt with Llama3 running on RPP.

        Args:
            kwargs (`dict`):
                The input information from user
        """
        prompt = kwargs.get('prompt')
        tools = kwargs.get('tools')
        self.stop_infer = False
        self.prefill_time = 0
        self.decode_time = 0
        self.total_time = 0              
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.str_tokens = ''
        self.cached_token.clear()
        prefilling_start_time = time.time()
        self.prefill_start_time = time.perf_counter()
        self.all_count = 34
        self.cur_count = 0
        input_ids, original_input_id, attention_mask, position_ids = self._apply_chat_prompt(prompt, tools)
        if input_ids is None:
            return

        token = self._run_prefill_inference(input_ids, original_input_id, attention_mask, position_ids)
        if self.stop_infer:
            return
        prefilling_end_time = time.time()
        decoding_start_time = prefilling_end_time
        original_input_id = original_input_id.tolist()
        original_input_id.append(token.tolist()[0])
        start_count = len(original_input_id)

        if self.low_power:
            self._graph_set_power_mode(0)
        # for decoding
        self._warmup(len(original_input_id) * 64)
        
        # [STEP 2] KV quant & set params
        self._init_decode_dynamic_param(len(original_input_id))

        self.attention_mask[:, :, :, :(len(original_input_id) + 1)] = 0.0
        self.attention_mask[:, :, :, (len(original_input_id) + 1):] = self.decoding_mask_num
        self.attention_mask_bindings = self.attention_mask

        while token.tolist()[0] != self.tokenizer.eos_token_id and not self.stop_infer:
            if len(original_input_id) >= (self.target_len):
                break
            self.attention_mask_bindings[:, :, :, len(original_input_id)] = 0.0
            self.hidden_states_bindings = self._run_embed_tokens(token.tolist()[0])
            # self.hidden_states_bindings = self.embed_tokens(token).unsqueeze(0).detach().numpy()
            position_ids_decode = np.ones((1, 1)).astype(np.int64) * (len(original_input_id) - 1)
            self.gather_out_0, self.gather_out_1 = self.decode_rotary_emb.run(['cosine', 'sine'], {'position_ids': position_ids_decode})
            
            self._update_decode_dynamic_param(len(original_input_id))
            self._run_decode_inference(len(original_input_id))
            tmp_list = []
            for i in range(0, 4):
                mv = memoryview(self.decode_host_buffer_list[i])
                tmp_list.append(np.ndarray(shape=(len(mv),), dtype=np.float32, buffer=mv))
            # lm_head_outs = np.concatenate(tmp_list, axis=0).reshape(1, 1, -1)
            lm_head_outs = np.concatenate(tmp_list, axis=0).reshape(1, -1)
            # get the next token
            token = self._get_next_token(original_input_id, lm_head_outs)
            original_input_id.append(token.tolist()[0])

            if self.write_file:
                continue
            # get the empty list to optimize the token print logits
            token_id = token.tolist()[0]
            # print the token
            if token.tolist()[0] != self.tokenizer.eos_token_id:
                try:
                    self.cached_token.append(token_id)
                    out_str = self.tokenizer.decode(self.cached_token, skip_special_tokens=True)
                    if '�' not in out_str:
                        if self.print_tokens:
                            print(out_str, end='', flush=True)
                        self.str_tokens = self.str_tokens + out_str
                        self.cached_token.clear()
                except (UnicodeDecodeError,AttributeError) as e:
                    print(token)
                    print('except:', e)
                    print(self.tokenizer.decoder[token.tolist()[0]])
            else:
                self.cached_token.append(token_id)
                out_str = ''.join(
                    [o for o in self.tokenizer.decode(self.cached_token, skip_special_tokens=True) if o != '�'])
                if self.print_tokens:
                    print(out_str, end='', flush=True)
                self.str_tokens = self.str_tokens + out_str
                self.cached_token.clear()

        # because of last loop already load param_x, so must exec param_x in the end
        self._update_decode_dynamic_param(len(original_input_id), True)
        if self.low_power:
            self._graph_set_power_mode(1)
        self.prompt_tokens = start_count - 1
        self.completion_tokens = len(original_input_id) - self.prompt_tokens
        self.total_tokens = len(original_input_id)
        decode_token = len(original_input_id) - start_count
        decoding_end_time = time.time()
        self.decode_time = (decoding_end_time - decoding_start_time) * 1000
        self.prefill_time = (prefilling_end_time - prefilling_start_time) * 1000
        self.total_time = (decoding_end_time - prefilling_start_time) * 1000
        print()
        self.logger.info(f"prefilling duration: {self.prefill_time:.2f} ms.")
        self.logger.info(f"decoding   duration: {self.decode_time:.2f} ms. decoding tokens: {decode_token} tokens. {(decode_token / (self.decode_time / 1e3)):.2f} tokens per second.")
        self.logger.info(f"total      duration: {self.total_time:.2f} ms. total tokens: {decode_token + 1} tokens\n")
        if self.write_file:
            decoder_str = self.tokenizer.decode(original_input_id, skip_special_tokens=True)
            self.write_file.write(decoder_str)
            self.write_file.flush()
            self.logger.info("the inference result already write to the prompt_result.txt file.")
        if self.perf_mode == 2:
            return self.tokenizer.decode(original_input_id[start_count - 1:], skip_special_tokens=True)
        if self.perf_mode == 1:
            return original_input_id[start_count - 1:]
        
    async def rpp_inference_stream(self, **kwargs):
        r"""
        Inference the prompt with Llama3 running on RPP for async.

        Args:
            kwargs (`dict`):
                The input information from user
        """
        prompt = kwargs.get('prompt')
        tools = kwargs.get('tools')
        self.stop_infer = False
        self.prefill_time = 0
        self.decode_time = 0
        self.total_time = 0              
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.str_tokens = ''
        self.cached_token.clear()
        prefilling_start_time = time.time()
        self.prefill_start_time = time.perf_counter()
        self.all_count = 34
        self.cur_count = 0
        input_ids, original_input_id, attention_mask, position_ids = self._apply_chat_prompt(prompt, tools)
        if input_ids is None:
            return

        self.prompt_len = len(original_input_id)
        hidden_states = []
        current_states = None
        cached_states = None
        if self.low_power:
            self._graph_set_power_mode(0)
        for i in range(input_ids.shape[-1]):
            if i <= len(original_input_id):
                token_id = input_ids[0, i].tolist()
                current_states = self._run_embed_tokens(token_id)
                if i == len(original_input_id):
                    cached_states = current_states.copy()
            else:
                current_states = cached_states
            hidden_states.append(current_states)
        hidden_states = np.stack(hidden_states, axis=0)
        # hidden_states = self.embed_tokens(input_ids).detach().numpy()
        attention_mask = attention_mask.detach().numpy()
        position_ids = position_ids.detach().numpy()
        
        if len(original_input_id) < 1024:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['1024'].run(['cosine', 'sine'], {'position_ids': position_ids})
        elif len(original_input_id) < 4096:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['4096'].run(['cosine', 'sine'], {'position_ids': position_ids})
        else:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['8192'].run(['cosine', 'sine'], {'position_ids': position_ids})
        
        # run prefill decoders
        prefill_input = []
        prefill_input.append(hidden_states.ravel())
        prefill_input.append(gather_out_0.ravel())
        prefill_input.append(gather_out_1.ravel())
        prefill_input.append(attention_mask.ravel())
        self._copy_prefill_init_data(prefill_input)
        kv_scales = []
        self._warmup(len(original_input_id) + 1, address=[0x6000810, 0x6001810])
        for i in range(32):
            if self.stop_infer:
                return
            self._print_prefill_progress()
            output = self._run_prefill_decoders(i)
            if not output[0]:
                return
            kv_scales.append(output[1])
            kv_scales.append(output[2])
            yield None
        self._run_prefill_nor_head()
        self._print_prefill_progress()
        self._warmup(int(math.ceil((len(original_input_id) + 1) / 256)), address=[0x6000804, 0x6001804])
        self._warmup(32 * 2, address=[0x6000808, 0x6001808])
        self._warmup(0, address=[0x600080C, 0x600180C])
        # push the quant_scales
        self._run_kv_init(np.concatenate(kv_scales))
        if self.low_power:
            self._graph_set_power_mode(1)

        lm_head_outs = self.prefill_host_buffer_list[0].reshape(1, -1)
        token = self._get_next_token(original_input_id, lm_head_outs)
        self._print_prefill_progress()
        if self.write_file:
            print("inference......, the result will write to the prompt_result.txt file.")
        else:
            try:
                self.cached_token.append(token.tolist()[0])
                out_str = self.tokenizer.decode(self.cached_token, skip_special_tokens=True)
                if '�' not in out_str:
                    if self.print_tokens:
                        print(out_str, end='', flush=True)
                    self.str_tokens = self.str_tokens + out_str
                    self.cached_token.clear()
                    yield self.str_tokens
            except (UnicodeDecodeError,AttributeError) as e:
                print(token)
                print('except:', e)
                print(self.tokenizer.decoder[token.tolist()[0]])
                
        if self.stop_infer:
            return
        prefilling_end_time = time.time()
        decoding_start_time = prefilling_end_time
        original_input_id = original_input_id.tolist()
        original_input_id.append(token.tolist()[0])
        start_count = len(original_input_id)

        if self.low_power:
            self._graph_set_power_mode(0)
        # for decoding
        self._warmup(len(original_input_id) * 64)
        
        # [STEP 2] KV quant & set params
        self._init_decode_dynamic_param(len(original_input_id))

        self.attention_mask[:, :, :, :(len(original_input_id) + 1)] = 0.0
        self.attention_mask[:, :, :, (len(original_input_id) + 1):] = self.decoding_mask_num
        self.attention_mask_bindings = self.attention_mask

        while token.tolist()[0] != self.tokenizer.eos_token_id and not self.stop_infer:
            if len(original_input_id) >= (self.target_len):
                break
            
            self.attention_mask_bindings[:, :, :, len(original_input_id)] = 0.0
            self.hidden_states_bindings = self._run_embed_tokens(token.tolist()[0])
            # self.hidden_states_bindings = self.embed_tokens(token).unsqueeze(0).detach().numpy()
            position_ids_decode = np.ones((1, 1)).astype(np.int64) * (len(original_input_id) - 1)
            self.gather_out_0, self.gather_out_1 = self.decode_rotary_emb.run(['cosine', 'sine'], {'position_ids': position_ids_decode})
            
            self._update_decode_dynamic_param(len(original_input_id))
            self._run_decode_inference(len(original_input_id))
            tmp_list = []
            for i in range(0, 4):
                mv = memoryview(self.decode_host_buffer_list[i])
                tmp_list.append(np.ndarray(shape=(len(mv),), dtype=np.float32, buffer=mv))
            # lm_head_outs = np.concatenate(tmp_list, axis=0).reshape(1, 1, -1)
            lm_head_outs = np.concatenate(tmp_list, axis=0).reshape(1, -1)
            # get the next token
            token = self._get_next_token(original_input_id, lm_head_outs)
            original_input_id.append(token.tolist()[0])

            if self.write_file:
                continue
            # get the empty list to optimize the token print logits
            token_id = token.tolist()[0]
            # print the token
            if token.tolist()[0] != self.tokenizer.eos_token_id:
                try:
                    self.cached_token.append(token_id)
                    out_str = self.tokenizer.decode(self.cached_token, skip_special_tokens=True)
                    if '�' not in out_str:
                        if self.print_tokens:
                            print(out_str, end='', flush=True)
                        self.str_tokens = self.str_tokens + out_str
                        self.cached_token.clear()
                        yield out_str
                except (UnicodeDecodeError,AttributeError) as e:
                    print(token)
                    print('except:', e)
                    print(self.tokenizer.decoder[token.tolist()[0]])
            else:
                self.cached_token.append(token_id)
                out_str = ''.join(
                    [o for o in self.tokenizer.decode(self.cached_token, skip_special_tokens=True) if o != '�'])
                if self.print_tokens:
                    print(out_str, end='', flush=True)
                self.str_tokens = self.str_tokens + out_str
                self.cached_token.clear()
                yield out_str

        # because of last loop already load param_x, so must exec param_x in the end
        self._update_decode_dynamic_param(len(original_input_id), True)
        if self.low_power:
            self._graph_set_power_mode(1)
        self.prompt_tokens = start_count - 1
        self.completion_tokens = len(original_input_id) - self.prompt_tokens
        self.total_tokens = len(original_input_id)
        decode_token = len(original_input_id) - start_count
        decoding_end_time = time.time()
        self.decode_time = (decoding_end_time - decoding_start_time) * 1000
        self.prefill_time = (prefilling_end_time - prefilling_start_time) * 1000
        self.total_time = (decoding_end_time - prefilling_start_time) * 1000
        print()
        self.logger.info(f"prefilling duration: {self.prefill_time:.2f} ms.")
        self.logger.info(f"decoding   duration: {self.decode_time:.2f} ms. decoding tokens: {decode_token} tokens. {(decode_token / (self.decode_time / 1e3)):.2f} tokens per second.")
        self.logger.info(f"total      duration: {self.total_time:.2f} ms. total tokens: {decode_token + 1} tokens\n")
        if self.write_file:
            decoder_str = self.tokenizer.decode(original_input_id, skip_special_tokens=True)
            self.write_file.write(decoder_str)
            self.write_file.flush()
            self.logger.info("the inference result already write to the prompt_result.txt file.")