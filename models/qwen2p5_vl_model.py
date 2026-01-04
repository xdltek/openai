import time
import numpy as np
import torch
import time
import os
import warnings
import math
import onnxruntime
import json
from tqdm import tqdm
from typing import List, Tuple, Dict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import OrderedDict
warnings.filterwarnings("ignore")
from transformers import AutoProcessor
from commons import LlmBaseModel
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

HEAD_NUM = 28
VISION_LOOP = 64
VISION_HEAD_NUM = 32
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

class Qwen2p5VLModel(LlmBaseModel):
    r"""
    @class Qwen2p5VLModel
    @brief Inherit LlmBaseModel and implement the inference function of Qwen2.5 VL.
    """
    def __init__(self,
                 rpp_dir: str,
                 input_size: int=1024,
                 target_len: int=2048,
                 write_file: int=0,
                 low_power: int=0,
                 prefix: int=1,
                 perf_mode: int=1,
                 print_tokens: int=1):
        r"""
        Class init function, construct the Qwen2p5VLModel class and initialize the parameters.
        
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
        super(Qwen2p5VLModel, self).__init__(rpp_dir=rpp_dir,graph_engine=None, low_power=low_power)
        self.visual = Qwen2p5VisionModel(rpp_dir=rpp_dir,
                                         graph_engine=self.graph_engine,
                                         share_index=0,
                                         low_power=low_power,
                                         target_len=2048)
        self.model = Qwen2p5Model(rpp_dir=rpp_dir,
                                  graph_engine=self.graph_engine,
                                  input_size=1024,
                                  target_len=2048,
                                  write_file=write_file,
                                  low_power=low_power,
                                  prefix=prefix,
                                  perf_mode=perf_mode,
                                  shared_index=self.visual.share_index,
                                  print_tokens=print_tokens)
    
    def _build_cpu_models(self):
        r"""
        Build the other models which is running in cpu, include AutoProcessor
        """
        # for auto processor
        self.processor = AutoProcessor.from_pretrained(os.path.join(self.rpp_dir, 'model_files'), use_fast=True)
    
    def set_infer_params(self,
                         penalty : float=1.1,
                         top_k: int=40,
                         top_p: float=0.9,
                         temperature: float=0.2,
                         min_tokens_to_keep: int=1,
                         do_sample: int=1):
        r"""
        Set inference params when get tokens.
        
        Args:
            penalty(`float`): 
                default is 1.1
            top_k(`int`):
                default is 40
            top_p(`float`):
                default is 0.9
            temperature(`float`):
                default is 0.2
            min_tokens_to_keep(`int`):
                default is 1
            do_sample(`int`):
                default is 1
        """
        self.model.set_infer_params(penalty=penalty,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature,
                                    min_tokens_to_keep=min_tokens_to_keep,
                                    do_sample=do_sample)  
        
    def _process_mm_datas(self, prompt):
        r"""
        Process vision information from prompt, get the input ids
        
        Args:
            prompt(`str` or `list`):
                The prompt which you want to inference. Can be a file path (str) or a list of messages (list)
        """
        if isinstance(prompt, str):
            # If prompt is a string, treat it as a file path
            with open(prompt, 'r',  encoding="utf-8") as f:
                messages = json.load(f)
        elif isinstance(prompt, list):
            # If prompt is a list, use it directly as messages
            messages = prompt
        else:
            raise ValueError(f"prompt must be either a file path (str) or a list of messages (list), got {type(prompt)}")
            
        if len(messages) == 0:
            print("prompt is empty, please check prompt.json file...")
            return None    
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
        
    def rpp_inference(self, **kwargs):
        r"""
        Inference the prompt with Qwen2.5 VL running on RPP.

        Args:
            kwargs (`dict`):
                The input information from user
        """
        prompt = kwargs.get('prompt')
        inputs = self._process_mm_datas(prompt)
        if inputs is None:
            return
        input_ids = inputs.input_ids
        
        # Check if there are any images in the input
        n_image_tokens = (input_ids == self.visual.image_token_id).sum().item()
        
        if n_image_tokens > 0:
            # Process vision inputs - only call vision inference if there are images
            if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
                image_embeds = self.visual.rpp_inference(inputs=inputs)      
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                
                inputs_embeds = self.model.embed_tokens(input_ids)
                mask = input_ids == self.visual.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                
                self.model.rpp_inference(input_ids=input_ids, hidden_states=inputs_embeds.detach().numpy())
            else:
                # Images were in the message but couldn't be processed, fall back to text-only
                self.model.rpp_inference(input_ids=input_ids)
        else:
            # Text-only input, use regular text model inference
            self.model.rpp_inference(input_ids=input_ids)
        
        # IMPORTANT: Propagate tokens and response from underlying model to VL model
        # The underlying Qwen2p5Model sets these values, we need to copy them to the VL model
        model_str_tokens = getattr(self.model, 'str_tokens', '') or ""
        if model_str_tokens and not isinstance(model_str_tokens, str):
            model_str_tokens = str(model_str_tokens)
        
        self.str_tokens = model_str_tokens
        self.prompt_tokens = getattr(self.model, 'prompt_tokens', 0)
        self.completion_tokens = getattr(self.model, 'completion_tokens', 0)
        self.total_tokens = getattr(self.model, 'total_tokens', 0)
    
    async def rpp_inference_stream(self, **kwargs):
        r"""
        Inference the prompt with Qwen2.5 VL running on RPP for streaming.

        Args:
            kwargs (`dict`):
                The input information from user
        """
        import sys
        
        try:
            prompt = kwargs.get('prompt')
            inputs = self._process_mm_datas(prompt)
            if inputs is None:
                return
            
            input_ids = inputs.input_ids
            
            # Check if there are any images in the input
            n_image_tokens = (input_ids == self.visual.image_token_id).sum().item()
            
            # Prepare hidden states based on whether we have images
            if n_image_tokens > 0 and 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
                # Process vision inputs
                image_embeds = self.visual.rpp_inference(inputs=inputs)      
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                
                inputs_embeds = self.model.embed_tokens(input_ids)
                mask = input_ids == self.visual.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                hidden_states = inputs_embeds.detach().numpy()
            else:
                # Text-only or fallback: generate hidden_states from input_ids
                hidden_states = self.model.embed_tokens(input_ids).detach().numpy()
            
            # Stream tokens from underlying model
            async for token_text in self.model.rpp_inference_stream(input_ids=input_ids, hidden_states=hidden_states):
                if token_text:
                    yield token_text
            
            # Propagate tokens and response from underlying model to VL model after streaming completes
            self.str_tokens = getattr(self.model, 'str_tokens', '') or ""
            self.prompt_tokens = getattr(self.model, 'prompt_tokens', 0)
            self.completion_tokens = getattr(self.model, 'completion_tokens', 0)
            self.total_tokens = getattr(self.model, 'total_tokens', 0)
            
        except Exception as e:
            import traceback
            print(f"Error in VL model streaming: {e}", file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            # Re-raise to let the caller handle it
            raise
          
class Qwen2p5VisionModel(LlmBaseModel):
    def __init__(self,
                 rpp_dir: str,
                 graph_engine,
                 share_index: int,
                 low_power: int,
                 target_len: int=2048):
        r"""
        Class init function, construct the Qwen2p5VisionModel class and initialize the parameters.
        
        Args:
            rpp_dir (`str`):
                The graph model path
            graph_engine:
                The graph engine
            low_power (`int`):
                The flag of low power, 0-not low power, 1-low power
            target_len (`int`):
                The maximum total size supported by the model
        """
        super(Qwen2p5VisionModel, self).__init__(rpp_dir=rpp_dir,
                                                 graph_engine=graph_engine, 
                                                 low_power=low_power,
                                                 input_size=1024,
                                                 target_len=2048)
        self.window_size = 112
        self.spatial_merge_size = 2
        self.patch_size = 14
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.image_token_id = 151655
        self.share_index = share_index
        
    def _set_sub_graph_names(self):
        r"""
        Set the sub names for graph models
        """
        # vision
        self.vision_embed_patch_name = 'vision/embed/patch'
        self.vision_embed_rotary_name = 'vision/embed/rotary'
        self.vision_top_name = 'vision/blocks/top'
        self.vision_mid_name = 'vision/blocks/mid'
        self.vision_bot_name = 'vision/blocks/bot'
        self.vision_cu_seqlen_name = 'vision/blocks/cu_seqlen'
        self.vision_merger_name = 'vision/merger'
        
    def _run_engine_cpu(self,
                        engine,
                        input_bindings: dict={}):
        onnx_outputs = engine.run(None, input_bindings)
        return onnx_outputs

    def _build_cpu_models(self):
        r"""
        Build the other models which is running in cpu, include vision onnx, block onnx
        """
        model_dir = os.path.join(self.rpp_dir, 'vision_onnx')
        self.engines = dict()
        self.engines['vision'] = {}
        # for embed 
        onnx_file = os.path.join(model_dir, 'vision', 'patchEmbed.onnx')
        self.engines['vision']['patch_embed_cpu'] = onnxruntime.InferenceSession(onnx_file)
        
        onnx_file = os.path.join(model_dir, 'vision', 'rotaryEmbedding.onnx')
        self.engines['vision']['rotary_embed_cpu'] = onnxruntime.InferenceSession(onnx_file)
        
        # for blocks
        for i in tqdm(range(VISION_HEAD_NUM)):
            path = os.path.join(model_dir, 'vision', 'blocks', f'{i}', f'top.onnx')
            self.engines['vision'][f'top_{i}_cpu'] = onnxruntime.InferenceSession(path)
            
            path = os.path.join(model_dir, 'vision', 'blocks', f'{i}', f'mid.onnx')
            self.engines['vision'][f'mid_{i}_cpu'] = onnxruntime.InferenceSession(path)
            
            path = os.path.join(model_dir, 'vision', 'blocks', f'{i}', f'bot.onnx')
            self.engines['vision'][f'bot_{i}_cpu'] = onnxruntime.InferenceSession(path)
            
            path = os.path.join(model_dir, 'vision', 'blocks', f'{i}', f'block_{i}_cu_seqlens.onnx')
            self.engines['vision'][f'cu_seqlens_{i}_cpu'] = onnxruntime.InferenceSession(path)
        
        # for merger
        onnx_file = os.path.join(model_dir, 'vision', 'patchMerger.onnx')
        self.engines['vision']['patch_merger_cpu'] = onnxruntime.InferenceSession(onnx_file)
    
    def _allocate_buffers(self):
        r"""
        Get bindings and allocate the host buffers for graph model
        """
        # for vision
        self.vision_top_i_bindings_list = []
        self.vision_top_o_bindings_list = []
        for i in range(VISION_HEAD_NUM):
            vision_top_x: str = f'{self.vision_top_name}{i}'
            vision_top_i_bindings, vision_top_o_bindings = self._alloc_graph_io_params(vision_top_x, 3)
            self.vision_top_i_bindings_list.append(vision_top_i_bindings)
            self.vision_top_o_bindings_list.append(vision_top_o_bindings)
        
        self.vision_bot_i_bindings_list = []
        self.vision_bot_o_bindings_list = []
        for i in range(VISION_HEAD_NUM):
            vision_bot_x: str = f'{self.vision_bot_name}{i}'
            vision_bot_i_bindings, vision_bot_o_bindings = self._alloc_graph_io_params(vision_bot_x, 3)
            self.vision_bot_i_bindings_list.append(vision_bot_i_bindings)
            self.vision_bot_o_bindings_list.append(vision_bot_o_bindings)
        
    def _rpp_inference_vision_embed(self, hidden_states):
        r"""
        Do vision embed inference on cpu
        
        Args:
            hidden_states:
                The hidden states
        """ 
        output = self._run_engine_cpu(engine=self.engines['vision']['patch_embed_cpu'],
                                      input_bindings={'hidden_states':hidden_states})
        output = output[0]
          
        return output

    def _rpp_inference_vision_merger(self, hidden_states): 
        r"""
        Do vision merger inference on cpu
        
        Args:
            hidden_states:
                The hidden states
        """ 
        output = self._run_engine_cpu(engine=self.engines['vision']['patch_merger_cpu'],
                                      input_bindings={'hidden_states':hidden_states})
        output = output[0]
        return output

    def _rpp_inference_vision_block(self, hidden_states, cos, sin, cu_window_seqlens, cu_seqlens):
        r"""
        Do vision block inference on cpu and rpp
        """ 
        current_size = hidden_states.shape[0]       
        for loop in tqdm(range(VISION_HEAD_NUM)):
            # cu_seqlens
            if loop in [7,15,23,31]:
                tmp_cu_seqlens = cu_seqlens
            else:
                tmp_cu_seqlens = cu_window_seqlens
            where_output = self._run_engine_cpu(engine=self.engines['vision'][f'cu_seqlens_{loop}_cpu'],
                                                input_bindings={'cu_seqlens':tmp_cu_seqlens})
            where_output = where_output[0]
            
            current_len = hidden_states.shape[0]
            pad_len = self.target_len - current_len
            padd_hidden_states = np.pad(hidden_states, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            padd_cos = np.pad(cos, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            padd_sin = np.pad(sin, ((0, pad_len), (0, 0)), mode='constant', constant_values=0) 
            # top
            out_266_list = []
            out_267_list = []
            out_2_list = []
            hidden_list = []
            for i in range(VISION_LOOP):
                start = i * 32
                end = start + 32
                hidden_chunk = padd_hidden_states[start:end, :]
                cos_chunk = padd_cos[start:end, :]
                sin_chunk = padd_sin[start:end, :]
                
                vision_top_name = f'{self.vision_top_name}{loop}'
                self.graph_engine.copy_fp32_from_numpy(self.vision_top_i_bindings_list[loop]['hidden_states'], hidden_chunk.ravel())
                self.graph_engine.copy_fp32_from_numpy(self.vision_top_i_bindings_list[loop]['position_embeddings'], cos_chunk.ravel())
                self.graph_engine.copy_fp32_from_numpy(self.vision_top_i_bindings_list[loop]['sin'], sin_chunk.ravel())
                self._graph_execute_sync(vision_top_name)
                out_266 = self.graph_engine.save_fp32_to_numpy(self.vision_top_o_bindings_list[loop]['/attn/Unsqueeze_266_output_0']).reshape(1, 16, 32, 80)
                out_267 = self.graph_engine.save_fp32_to_numpy(self.vision_top_o_bindings_list[loop]['/attn/Unsqueeze_267_output_0']).reshape(1, 16, 32, 80)
                out_2 = self.graph_engine.save_fp32_to_numpy(self.vision_top_o_bindings_list[loop]['/attn/Squeeze_2_output_0']).reshape(32, 16, 80)
                
                out_266_list.append(out_266)
                out_267_list.append(out_267)
                out_2_list.append(out_2)
                hidden_list.append(hidden_chunk)
                
                # top_outs = self._run_engine_cpu(engine=self.engines['vision'][f'top_{loop}_cpu'],
                #                                 input_bindings={'hidden_states':hidden_chunk,
                #                                                 'position_embeddings':cos_chunk,
                #                                                 'sin':sin_chunk})
                # out_266_list.append(top_outs[0])
                # out_267_list.append(top_outs[1])
                # out_2_list.append(top_outs[2])
                # hidden_list.append(hidden_chunk)
            
            input_267 = np.concatenate(out_267_list, axis=2)
            squ_input_2 = np.concatenate(out_2_list, axis=0)
            
            # mid and bot
            output_list = []
            for i in range(VISION_LOOP):
                start = i * 32
                end = start + 32
                where_output_chunk = where_output[:, start:end, :]
                
                mid_outs = self._run_engine_cpu(engine=self.engines['vision'][f'mid_{loop}_cpu'],
                                                input_bindings={'/attn/Unsqueeze_266_output_0':out_266_list[i],
                                                '/attn/Unsqueeze_267_output_0':input_267,
                                                '/attn/Where_72_output_0':where_output_chunk,
                                                '/attn/Squeeze_2_output_0':squ_input_2})
                transpose_5 = mid_outs[0]


                vision_bot_name = f'{self.vision_bot_name}{loop}'
                self.graph_engine.copy_fp32_from_numpy(self.vision_bot_i_bindings_list[loop]['hidden_states'], hidden_list[i].ravel())
                self.graph_engine.copy_fp32_from_numpy(self.vision_bot_i_bindings_list[loop]['/attn/Transpose_5_output_0'], transpose_5.ravel())
                self._graph_execute_sync(vision_bot_name)
                output_chunk = self.graph_engine.save_fp32_to_numpy(self.vision_bot_o_bindings_list[loop]['output']).reshape(32, 1280)
                 
                # bot_outs = self._run_engine_cpu(engine=self.engines['vision'][f'bot_{loop}_cpu'],
                #                             input_bindings={'hidden_states':hidden_list[i],
                #                             '/attn/Transpose_5_output_0':transpose_5})
                # output_chunk = bot_outs[0]
                
                output_list.append(output_chunk)
            hidden_states_all = np.concatenate(output_list, axis=0)
            hidden_states = hidden_states_all[:current_size, :]
        return hidden_states
    
    def rpp_inference(self, **kwargs):
        r"""
        Inference the prompt with Llama3 running on RPP.

        Args:
            kwargs (`dict`):
                The input information from user
        """
        # process vision datas
        inputs = kwargs.get('inputs')
        self.logger.info("do qwen2.5_vl vision inference...")
        grid_thw = inputs['image_grid_thw']
        hidden_states = inputs['pixel_values']
        valid_hidden_states_size = hidden_states.shape[0]
        pad_len = self.target_len - valid_hidden_states_size
        padd_hidden_states = np.pad(hidden_states.numpy(), ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        hidden_states_np = self._rpp_inference_vision_embed(padd_hidden_states)
        rotary_pos_emb = self._rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self._get_window_index(grid_thw)

        hidden_states = torch.from_numpy(hidden_states_np)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len = valid_hidden_states_size
        reordered = hidden_states[:valid_hidden_states_size, :].reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        reordered = reordered[window_index, :, :]
        hidden_states = reordered.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states_out = self._rpp_inference_vision_block(hidden_states.numpy(), 
                                                             emb.cos().numpy(), 
                                                             emb.sin().numpy(),
                                                             cu_window_seqlens.to(torch.int64).numpy(), 
                                                             cu_seqlens.to(torch.int64).numpy())
        # hidden_states_out_zz = np.load("./data/hidden_states_out_zz.npy")
        # mse, error = MSE(hidden_states_out_zz, hidden_states_out)
        # print(f"mse: {mse}, error: {error}")
        
        padd_hidden_states = np.pad(hidden_states_out, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        hidden_states_out = self._rpp_inference_vision_merger(padd_hidden_states)
        
        reordered = hidden_states_out[:seq_len // self.spatial_merge_unit, :]
        # rest = hidden_states_out[seq_len // self.spatial_merge_unit:]

        reverse_indices = torch.argsort(window_index)
        reverse_indices_np = reverse_indices.cpu().numpy()
        reordered = reordered[reverse_indices_np, :]
        
        # hidden_states_out = torch.cat([torch.from_numpy(reordered), torch.from_numpy(rest)], dim=0)
        return torch.from_numpy(reordered)

    def _rot_pos_emb(self, grid_thw: torch):
        r'''
        Get rotary embed information, now rotary onnx runing on cpu
        
        Args:
            grid_thw(`Tensor`):
                The grid size, include t, h, w
        '''
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
          
        rotary_pos_emb_full = self._run_engine_cpu(engine=self.engines['vision']['rotary_embed_cpu'],
                                                   input_bindings={'max_grid_size':max_grid_size.numpy()})
        rotary_pos_emb_full = rotary_pos_emb_full[0]   
        
        rotary_pos_emb_full_torch = torch.from_numpy(rotary_pos_emb_full)
        rotary_pos_emb = rotary_pos_emb_full_torch[pos_ids].flatten(1)

        return rotary_pos_emb

    def _get_window_index(self, grid_thw):
        r'''
        Get the windows index
        '''
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

class Qwen2p5Model(LlmBaseModel):
    r"""
    @class Qwen2p5Model
    @brief Inherit LlmBaseModel and implement the inference function of Qwen2.5.
    """
    def __init__(self, 
                 rpp_dir: str,
                 graph_engine,
                 input_size: int,
                 target_len: int,
                 write_file: int=0,
                 low_power: int=0,
                 prefix: int=1,
                 perf_mode: int=1,
                 shared_index: int=0,
                 print_tokens: int=1):
        r"""
        Class init function, construct the Qwen2p5Model class and initialize the parameters.
        
        Args:
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
                                           graph_engine=graph_engine, 
                                           low_power=low_power,
                                           input_size=1024,
                                           target_len=2048)
        self.write_file = write_file
        self.perf_mode = perf_mode
        self.prefix = prefix
        self.share_index = shared_index
        self.print_tokens = print_tokens
        self.attention_mask = np.ones((1, 1, 1, self.target_len), dtype=np.float32)
        self.inf_num = -1 * math.exp(60)      
        # init prompt len
        self.prompt_len = -1

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
        self.prefill_top_name = 'prefill_decoders/top'
        self.prefill_mid_name = 'prefill_decoders/mid'
        self.prefill_bot_name = 'prefill_decoders/bot'
        self.prefill_rms_norm_name = 'prefill_others/rms_norms/rms_norm'
        self.prefill_rms_param_name = 'prefill_others/rms_norms/rms_params'
        self.prefill_lm_head_name  = 'prefill_others/lm_heads/lm_head'
        self.prefill_top_param_name = 'prefill_others/top_params'
        self.prefill_mid_param_name = 'prefill_others/mid_params'
        self.prefill_bot_param_name = 'prefill_others/bot_params'
        self.prefill_k_quant_name = 'prefill_others/k_quants/k_quant'
        self.prefill_v_quant_name = 'prefill_others/v_quants/v_quant'
        self.prefill_embed_name = 'prefill_others/embed_tokens'
        self.prefill_k_reformat_name = 'prefill_others/kv_reformats/k_reformat'
        self.prefill_v_reformat_name = 'prefill_others/kv_reformats/v_reformat'

        # 1oop2
        self.decode_decodes_name  = 'decode_decoders/decoder'
        self.decode_lm_head_name  = 'decode_others/lm_heads/lm_head'
        self.decode_rms_norm_name = 'decode_others/rms_norm'
        # self.decode_kv_name    = 'decode_others/kv_plugin'
        self.decode_param_name = 'decode_others/graph_params/graph_param_'
        self.decode_k_reformat_name = 'prefill_others/decode_kv_ref/k_dec_ref'
        self.decode_v_reformat_name = 'prefill_others/decode_kv_ref/v_dec_ref'
        
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
        rotary_emb_names = {
            1024: 'prefill_rotary_emb_1024.onnx',
            2048: 'prefill_rotary_emb_2048.onnx',
            4096: 'prefill_rotary_emb_4096.onnx',
            7168: 'prefill_rotary_emb_7168.onnx'
        }
        self.prefill_rotary_emb = {
            '1024': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', rotary_emb_names[1024])),
            '2048': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', rotary_emb_names[2048])),
            '4096': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', rotary_emb_names[4096])),
            '7168': onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', rotary_emb_names[7168]))
        }
        self.decode_rotary_emb = onnxruntime.InferenceSession(os.path.join(self.rpp_dir, 'model_files', 'decode_rotary_emb.onnx'))
        try:
            self.embed_tokens = torch.jit.load(os.path.join(self.rpp_dir, 'model_files', 'embed_tokens_jit.pt'), map_location='cpu').eval()
        except:
            self.embed_tokens = torch.load(os.path.join(self.rpp_dir, 'model_files', 'embed_tokens_jit.pt'), map_location='cpu').eval()
 
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
        if self.prompt_len <= 1024:
            mid_name = 'mid0_1024'
        elif self.prompt_len <= 4096:
            mid_name = 'mid0_4096'
        elif self.prompt_len <= 7168:
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
        if self.prompt_len <= 1024:
            decoders_midx: str = f'{self.prefill_mid_name}0_1024'
            mid_param_name = 'mid0_1024'
        elif self.prompt_len <= 4096:
            decoders_midx: str = f'{self.prefill_mid_name}0_4096'
            mid_param_name = 'mid0_4096'
        elif self.prompt_len <= 7168:
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
        if self.prompt_len <= 1024:
            k_name = f'{self.prefill_k_reformat_name}_1024'
            v_name = f'{self.prefill_v_reformat_name}_1024'
        elif self.prompt_len <= 4096:
            k_name = f'{self.prefill_k_reformat_name}_4096'
            v_name = f'{self.prefill_v_reformat_name}_4096'
        elif self.prompt_len <= 7168:
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
        prefill_range = self.prompt_len // 128 + 1
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
        
    def _run_prefill_inference(self, hidden_states, original_input_ids, attention_mask, position_ids, ppl_flag=False):
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
        self.prompt_len = original_input_ids.shape[1]
        if self.low_power:
            self._graph_set_power_mode(0)
        # for i in range(input_ids.shape[-1]):
        #     if i <= self.prompt_len:
        #         token_id = input_ids[0, i].tolist()
        #         current_states = self._run_embed_tokens(token_id)
        #         if i == self.prompt_len:
        #             cached_states = current_states.copy()
        #     else:
        #         current_states = cached_states
        #     hidden_states.append(current_states)
        # hidden_states = np.stack(hidden_states, axis=0)

        attention_mask = attention_mask.detach().numpy()
        position_ids = position_ids.detach().numpy()[None, ...]
        if self.prompt_length <= 1024:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['1024'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})
        elif self.prompt_length <= 4096:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['4096'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})
        elif self.prompt_length <= 7168:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['7168'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})
        else:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['8192'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})

        # run prefill decoders
        prefill_input = []
        prefill_input.append(hidden_states.ravel())
        prefill_input.append(gather_out_0.ravel())
        prefill_input.append(gather_out_1.ravel())
        prefill_input.append(attention_mask.ravel())
        self._copy_prefill_init_data(prefill_input)
        # return_tuple = []
        # warmup for decoding stage reformat plug-in
        self._warmup((self.prompt_len + 1), address=[0x6000810, 0x6001810])
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
        self._warmup(int(math.ceil((self.prompt_len + 1) / 256)), address=[0x6000804, 0x6001804])
        self._warmup(HEAD_NUM * 2, address=[0x6000808, 0x6001808])
        self._warmup(0, address=[0x600080C, 0x600180C])
        # self._push_KV_for_decoding(np.array(return_tuple).reshape(HEAD_NUM, 2, -1, 4, 128, 128).transpose(0, 1, 3, 2, 4, 5).reshape(HEAD_NUM, 2, 4, -1, 128), self.prompt_len + 1)
        # get the quant_scales
        # self._warmup(int(math.ceil((self.prompt_len + 1) / 256)), address=[0x6000804, 0x6001804])
        # self._warmup(HEAD_NUM * 2, address=[0x6000808, 0x6001808])
        # self._warmup(0, address=[0x600080C, 0x600180C])
        # quant_scales = self._run_prefill_kv_quant(np.array(return_tuple).reshape(28, 2, -1, 4, 128, 128), self.prompt_len + 1)
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
    
    def _preprocess_prompt_for_vl(self, input_ids, hidden_states):
        r"""
        Tokenizer prompt information

        Args:
            input_ids (`Tensor`):
                The input ids for prompt
            hidden_states (`Tensor`):
                The hidden states
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
        input_ids = torch.Tensor(input_ids).to(torch.int64) 
        if input_ids.shape[-1] > 7168:
            input_size = 8192
        elif input_ids.shape[-1] > 4096:
            input_size = 7168
        elif input_ids.shape[-1] > 1024:
            input_size = 4096
        else:
            input_size = 1024

        # get real prompt_length, which records the total number of tokens (prompt + answer)
        input_len = input_ids.shape[-1]
        self.prompt_length = input_len

        # get the right padding number
        right_padding = input_size - input_len
        if right_padding <= 0:
            raise RuntimeError(f'Currently the prompt token length need <= {self.input_size}, your prompt token length is {input_ids.shape[-1]}')

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
        # input_ids = input_ids.squeeze()
        
        tmp_array = np.zeros((hidden_states.shape[0], input_size-hidden_states.shape[1], hidden_states.shape[2]), dtype=np.float32)
        pad_hidden_states = np.concatenate([hidden_states, tmp_array], axis=1)
        self.logger.info(f"original input_id size is {input_ids.shape[-1]}")
        return input_ids_padding, input_ids, attention_mask, position_ids, pad_hidden_states
    
    def rpp_inference(self, **kwargs):
        r"""
        Inference the prompt with Llama3 running on RPP.

        Args:
            kwargs (`dict`):
                The input information from user
        """
        input_ids = kwargs.get('input_ids')
        hidden_states = kwargs.get('hidden_states')
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

        input_ids, original_input_id, attention_mask, position_ids, pad_hidden_states = self._preprocess_prompt_for_vl(input_ids, hidden_states)
        if input_ids is None:
            return
        
        token = self._run_prefill_inference(pad_hidden_states, original_input_id, attention_mask, position_ids)
        if self.stop_infer:
            return
        
        prefilling_end_time = time.time()
        decoding_start_time = prefilling_end_time
        original_input_id = original_input_id.tolist()[0]
        original_input_id.append(token.tolist()[0])
        first_loop_len = self.input_size
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
            position_ids_decode = (np.ones((1, 1)).astype(np.int64) * (len(original_input_id) - 1))[None, ...]
            self.gather_out_0, self.gather_out_1 = self.decode_rotary_emb.run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                              {'position_ids': position_ids_decode})
            
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

            first_loop_len = first_loop_len + 1
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
        input_ids = kwargs.get('input_ids')
        hidden_states = kwargs.get('hidden_states')
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

        input_ids, original_input_id, attention_mask, position_ids, pad_hidden_states = self._preprocess_prompt_for_vl(input_ids, hidden_states)
        if input_ids is None:
            return
        
        self.prompt_len = original_input_id.shape[1]
        if self.low_power:
            self._graph_set_power_mode(0)
        # for i in range(input_ids.shape[-1]):
        #     if i <= self.prompt_len:
        #         token_id = input_ids[0, i].tolist()
        #         current_states = self._run_embed_tokens(token_id)
        #         if i == self.prompt_len:
        #             cached_states = current_states.copy()
        #     else:
        #         current_states = cached_states
        #     hidden_states.append(current_states)
        # hidden_states = np.stack(hidden_states, axis=0)

        attention_mask = attention_mask.detach().numpy()
        position_ids = position_ids.detach().numpy()[None, ...]
        if self.prompt_length <= 1024:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['1024'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})
        elif self.prompt_length <= 4096:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['4096'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})
        elif self.prompt_length <= 7168:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['7168'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})
        else:
            gather_out_0, gather_out_1 = self.prefill_rotary_emb['8192'].run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                             {'position_ids': position_ids})

        # run prefill decoders
        prefill_input = []
        prefill_input.append(hidden_states.ravel())
        prefill_input.append(gather_out_0.ravel())
        prefill_input.append(gather_out_1.ravel())
        prefill_input.append(attention_mask.ravel())
        self._copy_prefill_init_data(prefill_input)
        # return_tuple = []
        # warmup for decoding stage reformat plug-in
        self._warmup((self.prompt_len + 1), address=[0x6000810, 0x6001810])
        for i in range(HEAD_NUM):
            if self.stop_infer:
                return
            self._print_prefill_progress()
            self._run_prefill_decoders(i)
            yield None

        self._run_prefill_nor_head()
        self._print_prefill_progress()

        # push the BF16 KV to Decoding Stage decoders
        self._warmup(int(math.ceil((self.prompt_len + 1) / 256)), address=[0x6000804, 0x6001804])
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
        original_input_id = original_input_id.tolist()[0]
        original_input_id.append(token.tolist()[0])
        first_loop_len = self.input_size
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
            position_ids_decode = (np.ones((1, 1)).astype(np.int64) * (len(original_input_id) - 1))[None, ...]
            self.gather_out_0, self.gather_out_1 = self.decode_rotary_emb.run(['/self_attn/Unsqueeze_output_0', '/self_attn/Unsqueeze_1_output_0'], 
                                                                              {'position_ids': position_ids_decode})
            
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

            first_loop_len = first_loop_len + 1
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
        
    def _get_next_token(self,
                        input_ids: np.ndarray,
                        lm_head_input: np.ndarray) -> torch.Tensor:
        # logits = torch.Tensor(lm_head_input)
        # next_token_logits = logits[:, -1, :]
        next_token_logits = torch.Tensor(lm_head_input)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)[None, ...]
        score = torch.gather(next_token_logits, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        next_token_logits.scatter_(1, input_ids, score)
        if self.do_sample:
            sorted_logits, sorted_indices = torch.topk(next_token_logits, k=self.top_k, dim=-1, largest=True)
            sorted_logits = sorted_logits.flip(-1)
            sorted_indices = sorted_indices.flip(-1)
            sorted_logits = sorted_logits / (self.temperature + 1e-9)
            softmax_sorted_logits = sorted_logits.softmax(dim=-1)
            cumulative_probs = softmax_sorted_logits.cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
            softmax_sorted_logits[0, sorted_indices_to_remove.squeeze()] = 0.0

            next_tokens = torch.multinomial(softmax_sorted_logits, num_samples=1).to(torch.int64)
            next_tokens = sorted_indices[0, next_tokens].squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits,dim=-1)
        return next_tokens