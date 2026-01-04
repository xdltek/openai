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

HEAD_NUM = 28
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
UPDATE_PARAM_INIT = 8
UPDATE_PARAM_LOOP = 16

class Qwen2p5Model(LlmBaseModel):
    r"""
    @class Llama3Model
    @brief Inherit LlmBaseModel and implement the inference function of Qwen2.5.
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
        Class init function, construct the Qwen2p5Model class and initialize the parameters.
        
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
        super(Qwen2p5Model, self).__init__(rpp_dir=rpp_dir,
                                           graph_engine=None, 
                                           low_power=low_power,
                                           input_size=input_size,
                                           target_len=target_len)
        self.write_file = write_file
        self.perf_mode = perf_mode
        self.prefix = prefix
        self.attention_mask = np.ones((1, 1, 1, self.target_len), dtype=np.float32)
        self.inf_num = -1 * math.exp(60)      
        # init prompt len
        self.prompt_len = -1
        self.print_tokens = print_tokens

        # set default bindings
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
        # self.decode_kv_name    = 'loop2_others/kv_plugin'
        self.decode_param_name = 'loop2_others/graph_params/graph_param_'
        self.decode_k_reformat_name = 'loop1_others/decode_kv_ref/k_dec_ref'
        self.decode_v_reformat_name = 'loop1_others/decode_kv_ref/v_dec_ref'
        
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
            self.decode_k_reformat_name:self._share_index_increment(),
            self.decode_v_reformat_name:self._share_index_increment()
        }
        
        # allocate buffers
        self.decode_decoder_i_bindings_list = []
        decoders_x: str = f'{self.decode_decodes_name}{0}'
        self.decode_decoder_i_bindings = self._alloc_graph_io_params(decoders_x, 1)
        self.decode_decoder_i_bindings_list.append(self.decode_decoder_i_bindings)
        
        for i in range(1, HEAD_NUM):
            decoders_x: str = f'{self.decode_decodes_name}{i}'
            decode_decoder_i_bindings = self._alloc_graph_io_params(decoders_x, 1)
            self.decode_decoder_i_bindings_list.append(decode_decoder_i_bindings)
        # self.decode_kv_i_bindings = self._alloc_graph_io_params(self.decode_kv_name, 1)

        self.decode_lm_heads_o_bindings_list = [] 
        for i in range(0, 5):
            tmp_lm_head: str = f'{self.decode_lm_head_name}{i}'
            lm_head_out_bindings = self._alloc_graph_io_params(tmp_lm_head, 2)
            self.decode_lm_heads_o_bindings_list.append(lm_head_out_bindings)

        # prefill top in
        mul_ratio = self.input_size // 128
        decoders_top0: str = f'{self.prefill_top_name}0'
        self.prefill_decoder_top_i_0bindings = OrderedDict()
        prefill_decoder_top_i_0binding_0 = self.graph_engine.get_rpp_fw_in_one(decoders_top0,
                                                                                '/self_attn/Unsqueeze_output_0',
                                                                                1*1*128*128*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_top_i_0bindings['/self_attn/Unsqueeze_output_0'] = prefill_decoder_top_i_0binding_0
        prefill_decoder_top_i_0binding_1 = self.graph_engine.get_rpp_fw_in_one(decoders_top0,
                                                                                '/self_attn/Unsqueeze_1_output_0',
                                                                                1*1*128*128*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_top_i_0bindings['/self_attn/Unsqueeze_1_output_0'] = prefill_decoder_top_i_0binding_1
        prefill_decoder_top_i_0binding_2 = self.graph_engine.get_rpp_fw_in_one(decoders_top0,
                                                                                'hidden_in',
                                                                                1*128*3584*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_top_i_0bindings['hidden_in'] = prefill_decoder_top_i_0binding_2
        # prefill top out
        # self.prefill_decoder_top_o_0bindings = OrderedDict()
        # decoders_topx: str = f'{self.prefill_top_name}{0}'
        # prefill_decoder_top_o_binding_k = self.graph_engine.get_rpp_fw_out_one(decoders_topx,
        #                                                                     'past_key0',
        #                                                                     1*4*128*128*2*mul_ratio,
        #                                                                     self.IO_DATA_MODE_SHMGET)
        # self.prefill_decoder_top_o_0bindings['past_key0'] = prefill_decoder_top_o_binding_k
        # prefill_decoder_top_o_binding_v = self.graph_engine.get_rpp_fw_out_one(decoders_topx,
        #                                                                     'past_value0',
        #                                                                     1*4*128*128*2*mul_ratio,
        #                                                                     self.IO_DATA_MODE_SHMGET)
        # self.prefill_decoder_top_o_0bindings['past_value0'] = prefill_decoder_top_o_binding_v

        # prefill mid in/out
        self.prefill_decoder_mid_i_0bindings = OrderedDict()
        self.prefill_decoder_mid_o_0bindings = OrderedDict()
        decoders_mid0: str = f'{self.prefill_mid_name}0_{self.input_size}'
        prefill_decoder_mid_i_binding_0 = self.graph_engine.get_rpp_fw_in_one(decoders_mid0,
                                                                                '/self_attn/Slice_4_output_0',
                                                                                1*1*128*self.input_size*4*mul_ratio,
                                                                                self.IO_DATA_MODE_SHMGET)
        self.prefill_decoder_mid_i_0bindings['/self_attn/Slice_4_output_0'] = prefill_decoder_mid_i_binding_0
        
        # prefill top out
        self.prefill_decoder_top_o_27bindings = self._alloc_graph_io_params(f'{self.prefill_bot_name}27', 2, self.IO_DATA_MODE_NULL)
        
        # prefill rms
        self.prefill_rms_i_bindings, self.prefill_rms_o_bindings = self._alloc_graph_io_params(self.prefill_rms_norm_name, 3)
        

        # prefill lm_head
        self.prefill_lm_heads_o_bindings_list = OrderedDict()
        self.prefill_lm_heads_o_bindings_list['3'] = self.graph_engine.get_rpp_fw_out_one(f'{self.prefill_lm_head_name}{0}',
                                                                                        '3',
                                                                                        1*152064*4,
                                                                                        self.IO_DATA_MODE_SHMGET)
        # self.prefill_k_i_bindings, self.prefill_k_o_bindings = self._alloc_graph_io_params(f'{self.prefill_k_quant_name}0', 3)
        # self.prefill_v_i_bindings, self.prefill_v_o_bindings = self._alloc_graph_io_params(f'{self.prefill_v_quant_name}0', 3)
        
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
        try:
            self.embed_tokens = torch.jit.load(os.path.join(self.rpp_dir, 'model_files', 'embed_tokens.pt'), map_location='cpu').eval()
        except:
            self.embed_tokens = torch.load(os.path.join(self.rpp_dir, 'model_files', 'embed_tokens.pt'), map_location='cpu').eval()
 
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
        elif self.prompt_len < 7168:
            mid_name = 'mid0_7168'
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
        decoders_topx: str = f'{self.prefill_top_name}{loop}'
        self._graph_exec(decoders_topx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        if idx < range - 1:
            graph_top_param_x: str = f'{self.prefill_top_param_name}/top{loop}/graph_param_{idx + 1}'
            self._graph_load_sharedID(graph_top_param_x, self.global_share_id[self.prefill_top_param_name])
        self._graph_poll(decoders_topx)
             
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
            decoders_midx: str = f'{self.prefill_mid_name}0_1024'
            mid_param_name = 'mid0_1024'
        elif self.prompt_len < 4096:
            decoders_midx: str = f'{self.prefill_mid_name}0_4096'
            mid_param_name = 'mid0_4096'
        elif self.prompt_len < 7168:
            decoders_midx: str = f'{self.prefill_mid_name}0_7168'
            mid_param_name = 'mid0_7168'
        else:
            mid_param_name = 'mid0_8192'
            decoders_midx: str = f'{self.prefill_mid_name}0_8192'
        self._graph_exec(decoders_midx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        if idx < range - 1:
            graph_mid_param_x: str = f'{self.prefill_mid_param_name}/{mid_param_name}/graph_param_{idx + 1}'
            self._graph_load_sharedID(graph_mid_param_x, self.global_share_id[self.prefill_mid_param_name])
        self._graph_poll(decoders_midx)
    
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
        decoders_botx: str = f'{self.prefill_bot_name}{loop}'
        self._graph_exec(decoders_botx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        if idx < range - 1:
            graph_bot_param_x: str = f'{self.prefill_bot_param_name}/bot{loop}/graph_param_{idx + 1}'
            self._graph_load_sharedID(graph_bot_param_x, self.global_share_id[self.prefill_bot_param_name])
        self._graph_poll(decoders_botx)
        
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
        elif self.prompt_len < 7168:
            k_name = f'{self.prefill_k_reformat_name}_7168'
            v_name = f'{self.prefill_v_reformat_name}_7168'
        else:
            k_name = f'{self.prefill_k_reformat_name}_8192'
            v_name = f'{self.prefill_v_reformat_name}_8192'
        self._graph_execute_sync(k_name)
        self._graph_execute_sync(v_name)
    
    def _run_decoding_kv_reformat(self, index):
        r"""
        Reformat kv to decode
        
        Args:
            index('int'):
                Current index for decode
        """  
        k_name = f'{self.decode_k_reformat_name}_{index}'
        v_name = f'{self.decode_v_reformat_name}_{index}'

        repeats = self.target_len // 256

        graph_k_param_x = f'{k_name}/graph_param_0'
        self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.decode_k_reformat_name])
        for i in range(repeats):
            self._graph_execute_sync(graph_k_param_x)
            self._graph_exec(k_name, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if i < repeats - 1:
                graph_k_param_x = f'{k_name}/graph_param_{i + 1}'
                self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.decode_k_reformat_name])
            self._graph_poll(k_name)
        
        graph_v_param_x = f'{v_name}/graph_param_0'
        self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.decode_v_reformat_name])
        for i in range(repeats):
            self._graph_execute_sync(graph_v_param_x)
            self._graph_exec(v_name, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if i < repeats - 1:
                graph_v_param_x = f'{v_name}/graph_param_{i + 1}'
                self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.decode_v_reformat_name])
            self._graph_poll(v_name)

    def _run_prefill_decoders(self, loop):
        r"""
        Do prefill inference on rpp
        
        Args:
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

        # warmup for Ddecoding stage reformat plug-in
        self._warmup(0, address=[0x6000814, 0x6001814])
        self._warmup(0, address=[0x6000818, 0x6001818])
        self._run_decoding_kv_reformat(loop)
    
        mid_range = prefill_range * 2 if self.prompt_len > 7168 else prefill_range
        for idx in range(mid_range):
            self._update_prefill_mid_param(0, idx)
            self._run_prefill_mid(0, idx, mid_range)

        # for idx in range(prefill_range):
        #     self._update_prefill_bot_param(loop, idx)
        #     if idx < prefill_range - 1:
        #         self._run_prefill_bot(loop, idx, prefill_range)
        #     else:
        #         decoders_botx: str = f'{self.prefill_bot_name}{loop}'
        #         self._graph_exec(decoders_botx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)

        # N = int(math.ceil((self.prompt_len + 1) / 256) * 256)
        # past_k  = self.graph_engine.save_int16_to_int16_numpy(self.prefill_decoder_top_o_0bindings['past_key0'].tensor_haddr,
        #                                                        self.prefill_decoder_top_o_0bindings['past_key0'].tensor_daddr,
        #                                                        N*4*128*2,
        #                                                        self.prefill_decoder_top_o_0bindings['past_key0'].mapped)
        # past_v  = self.graph_engine.save_int16_to_int16_numpy(self.prefill_decoder_top_o_0bindings['past_value0'].tensor_haddr,
        #                                                        self.prefill_decoder_top_o_0bindings['past_value0'].tensor_daddr,
        #                                                        N*4*128*2,
        #                                                        self.prefill_decoder_top_o_0bindings['past_value0'].mapped)
        # init_tuple = ((past_k.reshape(1, -1, 4, 128, 128).transpose(0, 2, 1, 3, 4).reshape(1, 4, N, 128),) + 
        #                (past_v.reshape(1, -1, 4, 128, 128).transpose(0, 2, 1, 3, 4).reshape(1, 4, N, 128),))
        # init_tuple[0][:, :, (self.prompt_len + 1):, :] = .0
        # init_tuple[1][:, :, (self.prompt_len + 1):, :] = .0
        # self._graph_poll(decoders_botx)
        for idx in range(prefill_range):
            self._update_prefill_bot_param(loop, idx)
            decoders_botx: str = f'{self.prefill_bot_name}{loop}'
            self._graph_exec(decoders_botx, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if idx < prefill_range - 1:
                graph_bot_param_x: str = f'{self.prefill_bot_param_name}/bot{loop}/graph_param_{idx + 1}'
                self._graph_load_sharedID(graph_bot_param_x, self.global_share_id[self.prefill_bot_param_name])
            self._graph_poll(decoders_botx)
        return True
    
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
                                                               N*4*128*2,
                                                               self.prefill_decoder_top_o_0bindings[top_out_bindings_name].mapped)
        
        return top_out

    def _run_prefill_kv_quant(self, kv_arrays: Tuple[Tuple[np.ndarray]], token_length: int) -> Tuple[np.ndarray]:
        r"""
        Old interface, not use, do kv quantization
        
        Args:
            kv_arrays (`Tuple`): 
                Kv information
            token_length (`int`): 
                Current tokens length
        """ 
        kv_arrays = kv_arrays.transpose(0, 1, 3, 2, 4, 5).reshape(28, 2, 4, -1, 128)
        kv_arrays[:, :, :, token_length:, :] = 0.
        kv_arrays = kv_arrays.reshape(28, 2, 4, -1, 128, 128).transpose(0, 1, 3, 2, 4, 5)
        kv_scales = []
           
        for idx in range(kv_arrays.shape[0]):
            tmp_quant_k: str = f'{self.prefill_k_quant_name}{idx}'
            tmp_quant_v: str = f'{self.prefill_v_quant_name}{idx}'
            k_data = kv_arrays[idx][0]
            v_data = kv_arrays[idx][1]
            self.graph_engine.copy_data_to_device(k_data.ravel(), k_data.nbytes, self.prefill_k_i_bindings['input'].tensor_daddr)
            self.graph_engine.copy_data_to_device(v_data.ravel(), v_data.nbytes, self.prefill_v_i_bindings['input'].tensor_daddr)
            
            k_scales = []
            v_scales = []
            graph_k_param_x = f'{tmp_quant_k}/graph_param_0'
            self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.prefill_k_quant_name])
            for i in range(k_data.shape[0]//2):
                self._graph_execute_sync(graph_k_param_x)
                self._graph_exec(tmp_quant_k, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
                if i < k_data.shape[0]//2 - 1:
                    graph_k_param_x = f'{tmp_quant_k}/graph_param_{i + 1}'
                    self._graph_load_sharedID(graph_k_param_x, self.global_share_id[self.prefill_k_quant_name])
                self._graph_poll(tmp_quant_k)
                tmp_scales = self.graph_engine.save_int16_to_int16_numpy(self.prefill_k_o_bindings['output0'])
                k_scales.append(tmp_scales)
            
            graph_v_param_x = f'{tmp_quant_v}/graph_param_0'
            self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.prefill_v_quant_name])
            for j in range(k_data.shape[0]//2):
                self._graph_execute_sync(graph_v_param_x)
                self._graph_exec(tmp_quant_v, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
                if j < k_data.shape[0]//2 - 1:
                    graph_v_param_x = f'{tmp_quant_v}/graph_param_{j + 1}'
                    self._graph_load_sharedID(graph_v_param_x, self.global_share_id[self.prefill_v_quant_name])
                self._graph_poll(tmp_quant_v)
                tmp_scales = self.graph_engine.save_int16_to_int16_numpy(self.prefill_v_o_bindings['output0'])
                v_scales.append(tmp_scales)

            k_scales = np.concatenate(k_scales, axis=0)
            v_scales = np.concatenate(v_scales, axis=0)
            pad_factor = self.target_len - k_scales.shape[0]
            k_scales = np.pad(k_scales, ((0, pad_factor)), 'constant', constant_values=0)
            v_scales = np.pad(v_scales, ((0, pad_factor)), 'constant', constant_values=0)

            kv_scales.append(k_scales)
            kv_scales.append(v_scales)
        scales = np.concatenate(kv_scales)
        return scales
    
    def _run_embed_tokens(self, token_id):
        r"""
        Get embed data from device
        
        Args:
            token_id(`int`): 
                Current token number
        """ 
        offset = 3854 * 2
        daddr = self.embed_i_bindings['embedding_input'].tensor_daddr + offset * token_id
        haddr = self.embed_i_bindings['embedding_input'].tensor_haddr
        share = self.embed_i_bindings['embedding_input'].mapped
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
        # for i in range(rms_range):
        #     if i != rms_range -1:
        #         continue
        #     graph_rms_param_x: str = f'{self.prefill_rms_param_name}/graph_param_{i}'
        #     self._graph_load_sharedID(graph_rms_param_x, self.global_share_id[self.prefill_rms_param_name])
        #     self._graph_exec(graph_rms_param_x, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
        #     self._graph_poll(graph_rms_param_x)

        #     prefill_rms_norm_0: str = f'{self.prefill_rms_norm_name}'
        #     self._graph_execute_sync(prefill_rms_norm_0)
        #     for j in range(5):
        #         prefill_lm_head_x: str = f'{self.prefill_lm_head_name}{j}'
        #         self._graph_exec(prefill_lm_head_x, UPDATE_INTE + UPDATE_HEAD + UPDATE_TAIL)
        #         if j > 0:
        #             lm_out = self.graph_engine.save_fp32_to_numpy(self.prefill_lm_heads_o_bindings_list[j-1]['3'])
        #             self.prefill_host_buffer_list.append(lm_out)
        #         self._graph_poll(prefill_lm_head_x)
        #         if j == 4:
        #             lm_out = self.graph_engine.save_fp32_to_numpy(self.prefill_lm_heads_o_bindings_list[j]['3'])
        #             self.prefill_host_buffer_list.append(lm_out)
        self.prefill_host_buffer_list.clear()
        rms_norm_offset = (self.prompt_len-1) * 3584 * 4
        hidden_out = np.zeros((1,1,3584), np.float32)
        self.graph_engine.save_data_to_host(hidden_out, hidden_out.nbytes, self.prefill_decoder_top_o_27bindings['hidden_out'].tensor_daddr + rms_norm_offset)
        self.graph_engine.copy_data_to_device(hidden_out, hidden_out.nbytes, self.prefill_rms_i_bindings['input'].tensor_daddr)
        self._graph_execute_sync(self.prefill_rms_norm_name)
        for j in range(5):
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
    
    def _push_KV_for_decoding(self, kv_arrays: np.ndarray, token_length: int):
        r"""
        Push the Key & Value tensor to RPP DDR for decode decoders inference.

        Args:
            kv_arrays (`np.ndarray`):
                The Key & Value tensor in shape [HEAD_NUM, 2, 8, N*256, 128]
            token_length (`int`):
                The real token length of the prefill inference results
        """
        kv_arrays[:, :, :, token_length:, :] = 0.   # TODO: check this token_length or (token_length+1)
        pad_factor = [(0, 0), (0, 0), (0, 0), (0, self.target_len-kv_arrays.shape[3]), (0, 0)]
        kv_arrays = np.pad(kv_arrays, pad_factor, 'constant', constant_values=0)
        kv_arrays = kv_arrays.reshape(HEAD_NUM, 2, 4, -1, 256, 128)  # [HEAD_NUM, 2, 4, N, 256, 128]
        
        for i in range(HEAD_NUM):
            k_data = kv_arrays[i, 0, ...].transpose(1, 0, 3, 2)  # [N, 4, 128, 256]
            v_data = kv_arrays[i, 1, ...].transpose(1, 0, 2, 3)  # [N, 4, 256, 128]
            self.graph_engine.copy_data_to_device(k_data.ravel(), k_data.nbytes, self.decode_decoder_i_bindings_list[i]['past_key_in0'].tensor_daddr)
            self.graph_engine.copy_data_to_device(v_data.ravel(), v_data.nbytes, self.decode_decoder_i_bindings_list[i]['past_value_in0'].tensor_daddr)
        
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
        if self.low_power:
            self._graph_set_power_mode(0)
        # for i in range(input_ids.shape[-1]):
        #     if i <= len(original_input_ids):
        #         token_id = input_ids[0, i].tolist()
        #         current_states = self._run_embed_tokens(token_id)
        #         if i == len(original_input_ids):
        #             cached_states = current_states.copy()
        #     else:
        #         current_states = cached_states
        #     hidden_states.append(current_states)
        # hidden_states = np.stack(hidden_states, axis=0)
        hidden_states = self.embed_tokens(input_ids).detach().numpy()
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
        # return_tuple = []
        # warmup for decoding stage reformat plug-in
        self._warmup((len(original_input_ids) + 1), address=[0x6000810, 0x6001810])
        for i in range(HEAD_NUM):
            if self.stop_infer:
                return None
            self._print_prefill_progress()
            output = self._run_prefill_decoders(i)
            if not output:
                return None
            # return_tuple.append(output[1])

        self._run_prefill_nor_head()
        self._print_prefill_progress()

        # push the BF16 KV to Decoding Stage decoders
        self._warmup(int(math.ceil((len(original_input_ids) + 1) / 256)), address=[0x6000804, 0x6001804])
        self._warmup(HEAD_NUM * 2, address=[0x6000808, 0x6001808])
        self._warmup(0, address=[0x600080C, 0x600180C])
        # self._push_KV_for_decoding(np.array(return_tuple).reshape(HEAD_NUM, 2, -1, 4, 128, 128).transpose(0, 1, 3, 2, 4, 5).reshape(HEAD_NUM, 2, 4, -1, 128), len(original_input_ids) + 1)
        # get the quant_scales
        # self._warmup(int(math.ceil((len(original_input_ids) + 1) / 256)), address=[0x6000804, 0x6001804])
        # self._warmup(HEAD_NUM * 2, address=[0x6000808, 0x6001808])
        # self._warmup(0, address=[0x600080C, 0x600180C])
        # quant_scales = self._run_prefill_kv_quant(np.array(return_tuple).reshape(28, 2, -1, 4, 128, 128), len(original_input_ids) + 1)
        # push the quant_scales
        # if self.input_size != self.target_len:
        #     quant_scales = quant_scales.reshape(HEAD_NUM, 2, self.input_size)
        #     quant_scales = np.pad(quant_scales, ((0, 0), (0, 0), (0, self.target_len-self.input_size)), 'constant', constant_values=0).ravel()
        # self._run_kv_init(quant_scales)

        if self.low_power:
            self._graph_set_power_mode(1)

        # if self.prompt_len % 256 == 0:
        #     rms_range = self.prompt_len // 256
        # else:
        #     rms_range = self.prompt_len // 256 + 1
        # tmp_list = []
        # for rms_norm_idx in range(rms_range):
        #     if rms_norm_idx != (rms_range-1):
        #         continue
        #     tmp_np_array = []
        #     for lm_head_idx in range(5):
        #         # mv = memoryview(self.prefill_host_buffer_list[rms_norm_idx * 5 + lm_head_idx])
        #         mv = memoryview(self.prefill_host_buffer_list[lm_head_idx])
        #         tmp_np_array.append(np.ndarray(shape=(len(mv),), dtype=np.float32, buffer=mv).reshape(1, 256, -1))
        #     tmp_list.append(np.concatenate(tmp_np_array, axis=2))
        # _offset = self.prompt_len - (256*(rms_range-1)) - 1
        # lm_head_outs = np.concatenate(tmp_list, axis=1)[:, _offset, :]
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
        self.graph_engine.copy_fp32_from_numpy(self.decode_decoder_i_bindings['hidden_in'], self.hidden_states_bindings)
        self.graph_engine.copy_fp32_from_numpy(self.decode_decoder_i_bindings['attention_mask'], self.attention_mask_bindings)
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
        # nor
        graph_param_x: str = f'{self.decode_param_name}{loop + 1}'
        self._graph_exec(self.decode_rms_norm_name, UPDATE_TAIL + UPDATE_INTE)
        if loop + 1 < self.target_len:
            self._graph_load_sharedID(graph_param_x, self.global_share_id[self.decode_param_name])
        self._graph_poll(self.decode_rms_norm_name)
        # lm header
        self.decode_host_buffer_list.clear()
        for idx in range(5):
            graph_lm_head_x: str = f'{self.decode_lm_head_name}{idx}'
            self._graph_exec(graph_lm_head_x, UPDATE_HEAD + UPDATE_INTE + UPDATE_TAIL)
            if idx > 0:
                output = self.graph_engine.save_fp32_to_numpy(self.decode_lm_heads_o_bindings_list[idx-1]['3'])
                self.decode_host_buffer_list.append(output)
            self._graph_poll(graph_lm_head_x)
            if idx == 4:
                output = self.graph_engine.save_fp32_to_numpy(self.decode_lm_heads_o_bindings_list[idx]['3'])
                self.decode_host_buffer_list.append(output)
        return True

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
            conversation = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
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
            input_ids = torch.tensor([self.tokenizer.encode(prompt)])
            tmp_attention_mask = torch.full((1, input_ids.shape[-1]), 1, dtype=torch.int)
        
        if input_ids.shape[-1] >= 7168:
            input_size = 8192
        elif input_ids.shape[-1] >= 4096:
            input_size = 7168
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
        
        # padding the input_ids, 128009 is the magic number of LLAMA3, which is the total embedding space of LLAMA3 model
        fixed_tensor = torch.full((1, right_padding), 151643, dtype=torch.int)

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
    
    def _build_prompt(self,
                    query,
                    history=None,
                    system="You are a helpful assistant"):
        if history is None:
            history=[]

        # 包裹发言内容的token
        im_start,im_start_tokens='<|im_start|>',[self.tokenizer.all_special_ids[2]]
        im_end,im_end_tokens='<|im_end|>',[self.tokenizer.all_special_ids[0]]
        # 换行符token
        nl_tokens=self.tokenizer.encode("\n")

        # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
        def _tokenize_str(role,content): # 返回元组，下标0是文本，下标1是token ids
            return f"{role}\n{content}",self.tokenizer.encode(role)+nl_tokens+self.tokenizer.encode(content)
        
        # 剩余token数
        left_token_space=self.input_size

        # prompt头部: system发言
        system_text_part,system_tokens_part=_tokenize_str("system", system) # system_tokens_part -->    system\nYou are a helpful assistant.
        system_text=f'{im_start}{system_text_part}{im_end}'
        system_tokens=im_start_tokens+system_tokens_part+im_end_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
        left_token_space-=len(system_tokens)
        
        # prompt尾部: user发言和assistant引导
        query_text_part,query_tokens_part=_tokenize_str('user', query)
        query_tokens_prefix=nl_tokens+ im_start_tokens
        query_tokens_suffix=im_end_tokens+nl_tokens+im_start_tokens+self.tokenizer.encode('assistant')+nl_tokens
        if len(query_tokens_prefix)+len(query_tokens_part)+len(query_tokens_suffix)>left_token_space: # query太长截断
            query_token_len=left_token_space-len(query_tokens_prefix)-len(query_tokens_suffix)-2 # need padding and decode run, so -2
            query_tokens_part=query_tokens_part[:query_token_len]
            query_text_part=self.tokenizer.decode(query_tokens_part)
            print(f"your prompt is > {self.input_size - 1}, will truncate the prompt.")
        query_tokens=query_tokens_prefix+query_tokens_part+query_tokens_suffix
        query_text=f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"
        left_token_space-=len(query_tokens)
        
        # prompt腰部: 历史user+assitant对话
        history_text,history_tokens='',[]
        for hist_query,hist_response in reversed(history):    # 优先采用最近的对话历史
            hist_query_text,hist_query_tokens_part=_tokenize_str("user",hist_query) # user\n历史提问
            hist_response_text,hist_response_tokens_part=_tokenize_str("assistant",hist_response) # assistant\n历史回答
            # 生成本轮对话
            cur_history_tokens=nl_tokens+im_start_tokens+hist_query_tokens_part+im_end_tokens+nl_tokens+im_start_tokens+hist_response_tokens_part+im_end_tokens
            cur_history_text=f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
            # 储存多轮对话
            if len(cur_history_tokens)<=left_token_space:
                history_text=cur_history_text+history_text
                history_tokens=cur_history_tokens+history_tokens
                left_token_space-=len(cur_history_tokens)
            else:
                break 
                
        # 生成完整Prompt
        # prompt_str=f'{system_text}{history_text}{query_text}'
        # prompt_tokens=system_tokens+history_tokens+query_tokens
        input_ids = torch.tensor(system_tokens+history_tokens+query_tokens).reshape(1,-1)
        input_len = input_ids.shape[-1]
        
        input_size = self.input_size if input_ids.shape[-1] > 4095 else 4096
        input_size = input_size if input_ids.shape[-1] > 1023 else 1024
        
        # right_padding = left_token_space
        right_padding = input_size - input_len
        # padding the input_ids, 151643 is the magic number of LLAMA3, which is the total embedding space of LLAMA3 model
        fixed_tensor = torch.full((1, right_padding), 151643, dtype=torch.int)

        # concat the tensors & get the real input_ids
        input_ids_padding = torch.cat([input_ids, fixed_tensor], 1)

        # get the attention_mask
        attention_mask_left = torch.tensor(np.ones((1, input_len), dtype=np.float32))
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
        self.all_count = HEAD_NUM + 2
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

        # [STEP 2] KV quant & set params
        self._warmup(len(original_input_id) * 64)
        self._init_decode_dynamic_param(len(original_input_id))

        self.attention_mask[:, :, :, :(len(original_input_id) + 1)] = 0.0
        self.attention_mask[:, :, :, (len(original_input_id) + 1):] = self.inf_num
        self.attention_mask_bindings = self.attention_mask

        while token.tolist()[0] != self.tokenizer.eos_token_id and not self.stop_infer:
            if len(original_input_id) >= (self.target_len):
                break
            self.attention_mask_bindings[:, :, :, len(original_input_id)] = 0.0
            # self.hidden_states_bindings = self._run_embed_tokens(token.tolist()[0])
            self.hidden_states_bindings = self.embed_tokens(token).unsqueeze(0).detach().numpy()
            position_ids_decode = np.ones((1, 1)).astype(np.int64) * (len(original_input_id) - 1)
            self.gather_out_0, self.gather_out_1 = self.decode_rotary_emb.run(['cosine', 'sine'], {'position_ids': position_ids_decode})
            
            self._update_decode_dynamic_param(len(original_input_id))
            self._run_decode_inference(len(original_input_id))
            tmp_list = []
            for i in range(0, 5):
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
        Inference the prompt with Qwen2.5 running on RPP for async.

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
        self.all_count = HEAD_NUM + 2
        self.cur_count = 0

        input_ids, original_input_id, attention_mask, position_ids = self._apply_chat_prompt(prompt, tools)
        if input_ids is None:
            return
        
        self.prompt_len = len(original_input_id)
        hidden_states = []
        if self.low_power:
            self._graph_set_power_mode(0)
        # for i in range(input_ids.shape[-1]):
        #     if i <= len(original_input_ids):
        #         token_id = input_ids[0, i].tolist()
        #         current_states = self._run_embed_tokens(token_id)
        #         if i == len(original_input_ids):
        #             cached_states = current_states.copy()
        #     else:
        #         current_states = cached_states
        #     hidden_states.append(current_states)
        # hidden_states = np.stack(hidden_states, axis=0)
        hidden_states = self.embed_tokens(input_ids).detach().numpy()
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
        # return_tuple = []
        # warmup for decoding stage reformat plug-in
        self._warmup((len(original_input_id) + 1), address=[0x6000810, 0x6001810])
        for i in range(HEAD_NUM):
            if self.stop_infer:
                return
            self._print_prefill_progress()
            self._run_prefill_decoders(i)
            yield None
        self._run_prefill_nor_head()
        self._print_prefill_progress()

        # push the BF16 KV to Decoding Stage decoders
        self._warmup(int(math.ceil((len(original_input_id) + 1) / 256)), address=[0x6000804, 0x6001804])
        self._warmup(HEAD_NUM * 2, address=[0x6000808, 0x6001808])
        self._warmup(0, address=[0x600080C, 0x600180C])
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

        # [STEP 2] KV quant & set params
        self._warmup(len(original_input_id) * 64)
        self._init_decode_dynamic_param(len(original_input_id))

        self.attention_mask[:, :, :, :(len(original_input_id) + 1)] = 0.0
        self.attention_mask[:, :, :, (len(original_input_id) + 1):] = self.inf_num
        self.attention_mask_bindings = self.attention_mask

        while token.tolist()[0] != self.tokenizer.eos_token_id and not self.stop_infer:
            if len(original_input_id) >= (self.target_len):
                break
            
            self.attention_mask_bindings[:, :, :, len(original_input_id)] = 0.0
            # self.hidden_states_bindings = self._run_embed_tokens(token.tolist()[0])
            self.hidden_states_bindings = self.embed_tokens(token).unsqueeze(0).detach().numpy()
            position_ids_decode = np.ones((1, 1)).astype(np.int64) * (len(original_input_id) - 1)
            self.gather_out_0, self.gather_out_1 = self.decode_rotary_emb.run(['cosine', 'sine'], {'position_ids': position_ids_decode})
            
            self._update_decode_dynamic_param(len(original_input_id))
            self._run_decode_inference(len(original_input_id))
            tmp_list = []
            for i in range(0, 5):
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
