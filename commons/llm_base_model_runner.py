# -*- coding: utf-8 -*-
import os
import sys
import math
import time
import json
import torch
import warnings
import subprocess
import onnxruntime
import numpy as np
import shutil
import platform

from tqdm import tqdm
from loguru import logger
from collections import OrderedDict
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from functools import wraps

warnings.filterwarnings("ignore")
if platform.system().lower() == 'windows':
    sys.path.append('../lib/')
else:
    sys.path.append('/usr/local/rpp/lib')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pyfwgraphs as fw

def check_return(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if not result:
            print(">>>>>> {0} return {1}, program exited".format(func.__name__, result))
            self.delete_rpp_fws()
            raise SystemExit(1)
        return result
    return wrapper


class LlmBaseModel(object):
    r"""
    @class LlmBaseModel
    @brief The RPP LLM base model implementation, defines common interfaces, and subclasses inherit and implement these interfaces.
    """
    def __init__(self, 
                 rpp_dir: str,
                 graph_engine: None,
                 low_power: int=0,
                 input_size: int=2048,
                 target_len: int=2048):
        r"""
        Class init function, construct the class and initialize the parameters.
        
        Args:
            rpp_dir(`str`):
                The path of graph model
            graph_engine:
                The engine of graph
            low_power(`int`):
                The flag of low power, 0-not low power, 1-low power
            input_size(`int`):
                The maximum input size supported by the model
            target_len(`int`):
                The maximum total size supported by the model
        """
        super(LlmBaseModel, self).__init__()
        self.rpp_dir = rpp_dir
        self.graph_engine = graph_engine
        self.low_power = low_power
        self.target_len = target_len
        self.input_size = input_size
        self.share_index = 0
        self.stop_infer = False
        self.penalty = 1.1
        self.top_p = 0.9
        self.top_k = 40
        self.temperature = 0.2
        self.min_tokens_to_keep = 1
        self.do_sample = 1
        self.str_tokens = ''
        self.prefill_time = 0
        self.decode_time = 0
        self.total_time = 0              
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        self.IO_DATA_MODE_NULL = fw.IO_DATA_MODE_NULL
        self.IO_DATA_MODE_SHMGET = fw.IO_DATA_MODE_SHMGET
        self.IO_DATA_MODE_MALLOC = fw.IO_DATA_MODE_MALLOC
        self.IO_DATA_MODE_PHYSIC = fw.IO_DATA_MODE_PHYSIC

        self._init_logger()
        self._init()
    
    def _init_logger(self, log_lvl: str='INFO'):
        r"""
        Init logger, set the log format,level information.
        
        Args:
            log_lvl(`str`):
                Log level, the default is info
        """
        logger.configure(
            handlers=[],
            levels={},
            extra={},
            patcher=None,
            activation=None
        )

        log_format = "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
        logger.configure(
            handlers=[
                # log on console
                {
                    "sink": sys.stdout,
                    "level": log_lvl,
                    "format": log_format
                }
            ]
        )
        self.logger = logger
    
    def _show_system_info(self):
        r"""
        Show the details of the CPU.
        """
        if platform.system().lower() == 'windows':
            command = 'wmic cpu get /format:list'
            result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            for line in result.stdout:
                line = line.decode('utf-8', errors='ignore').strip()
                if line:
                    print(line)
        else:
            command = "lscpu"
            output = subprocess.check_output(command, shell=True, universal_newlines=True)
            index = output.index("Vulnerability")
            print(output[0 : index])
    
    def _volume(self, obj: List[int]) -> int:
        r"""
        Return the total volume(number of the elements) of the target tensor.
        
        Args:
            obj(`List[int]`):
                The list contains the dimensions of the target tensor
            vol(`int`):
                The total volume(number of the elements) of the target tensor
        """
        vol = 1
        for elem in obj:
            vol *= elem
        return vol
    
    def _check_mse(x, y):
        r"""
        Compare the mean squared error of inputs x and y.
        
        Args:
            x(`numpy`) 
                Inputs x numpy
            y(`numpy`) I
                nputs y numpy
        Returns: 
            mse, error
        """
        error = np.power(np.abs(x.ravel() - y.ravel()), 2).sum()
        base = np.power(y.ravel(), 2).sum()
        mse = error / base
        return mse, error
    
    def _check_sum(self, name: str):
        r"""
        Get the sum of outputs data form rpp device.
        
        Args:
            name(`str`):
                the sub graph name
        Returns:
            outs(`dict`):
                the sum value of output, key is output name, value is the sum
        """
        o_bindings_dict = self._alloc_graph_io_params(name, 2, self.IO_DATA_MODE_NULL)
        outs = {}  
        for key, value in o_bindings_dict.items():
            host_buf = np.zeros((value.tensor_size), np.int8)
            self.global_graphs.save_data_to_host(host_buf, host_buf.nbytes, value.tensor_daddr)
            sum_data = np.sum(host_buf.astype(np.int64))
            outs[key] = sum_data
            print(f'graph name: {name}, tensor name: {key}, sum: {sum_data}')
        return outs
    
    def _share_index_increment(self):
        r"""
        For dynamic graph, set shared index.
        """
        self.share_index += 1
        return self.share_index
    
    def _init_tokenizer(self):
        r"""
        Build the LLM tokenizer for encoding & decoding the prompts.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.rpp_dir, 'model_files'))
        
    def _build_rpp_models(self):
        r"""
        Build the LLM prefill & decode engines on RPP.
        """
        if self.graph_engine is None:
            self.graph_engine = fw.RppFwGraphs(self.rpp_dir)
            if self.low_power:
                self._graph_set_power_mode(0)
            load_graph_start = time.perf_counter()
            logger.info('loading graph models...............')
            self._graph_deserialize_all()
            load_graph_end = time.perf_counter()
            load_graph_cost = load_graph_end - load_graph_start
            logger.info(f"load all graph time = {load_graph_cost:.2f} seconds.")
        self._set_sub_graph_names()
        self._allocate_buffers()
        if self.low_power:
            self._graph_set_power_mode(1)
        
    def _alloc_graph_io_params(self, graph_name: str, ntype: int, mtype: int = 2):
        r"""
        Get the io informations for graph model

        Args:
            graph_name(`str`):
                The name of sub-graph
            ntype(`int`):
                1: inputs, 2: outputs, 3: inputs&outputs
            mtype(`int`): 
                Memory allocation method: 
                    IO_DATA_MODE_NULL   = 1 
                    IO_DATA_MODE_SHMGET = 2 
                    IO_DATA_MODE_MALLOC = 4
                    IO_DATA_MODE_PHYSIC = 8
        Returns:
            inputs and outputs bingdings 
        """
        i_bindings_dict = OrderedDict()
        o_bindings_dict = OrderedDict()
        # get in bindings
        if ntype == 1:
            i_bindings = self.graph_engine.get_rpp_fw_in(graph_name, mtype)
            for item_i in i_bindings:
                i_bindings_dict[item_i.tensor_name] = item_i
            return i_bindings_dict
        # get out bindings
        elif ntype == 2:
            o_bindings = self.graph_engine.get_rpp_fw_out(graph_name, mtype)
            for itme_o in o_bindings:
                o_bindings_dict[itme_o.tensor_name] = itme_o
            return o_bindings_dict
        # get in and out bindings
        else:
            i_bindings = self.graph_engine.get_rpp_fw_in(graph_name, mtype)
            for item_i in i_bindings:
                i_bindings_dict[item_i.tensor_name] = item_i

            o_bindings = self.graph_engine.get_rpp_fw_out(graph_name, mtype)
            for itme_o in o_bindings:
                o_bindings_dict[itme_o.tensor_name] = itme_o
            return i_bindings_dict, o_bindings_dict
        
    def _warmup(self, length, address = [0x6000800, 0x6001800]):
        r"""
            Set warmup information for device
            
        Args:
            length(`int`):
                the value writed to device
            address(`List`):
                the address of device
        """
        self.graph_engine.read_write_bar_2(length, address[0], False)
        self.graph_engine.read_write_bar_2(length, address[1], False)
    
    def _get_next_token(self,
                        input_ids: np.ndarray,
                        lm_head_input: np.ndarray) -> torch.Tensor:
        r"""
        Get token id from lm head output using top_k, top_p, temperature, penalty and so on.
        
        Args:
            input_ids(`np.ndarray`):
                The input ids
            lm_head_input(`np.ndarray`): 
                lm head outputs
        """
        # logits = torch.Tensor(lm_head_input)
        # next_token_logits = logits[:, -1, :]
        next_token_logits = torch.Tensor(lm_head_input)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
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

    def _init(self):
        r"""
        Init the models for RPP running, include build cpu engine and rpp engine
        """ 
        self._init_tokenizer()
        self._build_cpu_models()
        self._build_rpp_models()
        
    def _allocate_buffers(self):
        r"""
        Virtual function, Get bindings and allocate the host buffers for graph
        """
        
    def _set_sub_graph_names(self):
        r"""
        Virtual function, Set the sub names for graph models
        """
        
    def _build_cpu_models(self):
        r"""
        Virtual function, build the other models which is running in cpu.
        """
        
    def rpp_inference(self, **kwargs):
        r"""
        Virtual function, inference the prompt with LLM models running on RPP.
        
        Args:
            kwargs(`Dict`):
                The input prompt from user
        """
        raise NotImplementedError
    
    async def rpp_inference_stream(self, **kwargs):
        r"""
        Virtual function, Inference the prompt with LLM models running on RPP, yield result.
        
        Args:
            kwargs(`Dict`): 
                The input prompt from user
        """
    
    def stop_inference(self, bflag: bool):
        r"""
        Set flag to stop inference.
        
        Args:
            bflag(`bool`): 
                stop flag, True:stop infernce
        """
        self.stop_infer = bflag
    
    def delete_rpp_fws(self):
        r"""
        Delete graph engine to release resouce
        """
        if self.graph_engine:
            del self.graph_engine
            self.graph_engine = None
            logger.info('delete rpp fws instance.............')
     
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
        self.penalty = penalty
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        self.do_sample = do_sample
    
    @check_return
    def _graph_set_power_mode(self, mode):
        r"""
        Set low power mode.
        
        Args:
            mode:
                0-work mode, 1-low power mode
                
        Returns:
            True or False
        """
        return self.graph_engine.get_rpp_set_power_mode(mode)

    @check_return
    def _graph_deserialize(self, names, values):
        r"""
        Old interface, abandonment, deserialize graph model, load to ddr
        
        Args:
            names(`str`):
                The graph model path
            values:
                Graph flag
                
        Returns:
            True or False
        """
        return self.graph_engine.gen_rpp_fws(names, values)
    
    @check_return
    def _graph_deserialize_all(self):
        r"""
        Deserialize graph model, load to ddr
        
        Returns:
            True or False
        """
        return self.graph_engine.gen_rpp_fws()
  
    @check_return
    def _graph_load(self, name):
        r"""
        Load static sub model to device
        
        Args:
            name(`str`):
                The graph name
        Returns:
            True or False
        """
        return self.graph_engine.load_rpp_fw(name)
    
    @check_return
    def _graph_load_sharedID(self, name, sharedID):
        r"""
        Load dynamic sub model to device
        
        Args:
            name(`str`):
                The graph name
                
        Returns:
            True or False
        """
        return self.graph_engine.load_rpp_fw(name, sharedID)

    @check_return
    def _graph_execute_sync(self, names):
        r"""
        Run sub graph model on deivece
        
        Args:
            name(`str`):
                the graph name
                
        Returns:
            True or False
        """
        return self.graph_engine.exec_rpp_fw_sync(names)
    
    @check_return
    def _graph_exec(self, name, mode):
        r"""
        Exec sub graph model on deivece
        
        Args:
            name(`str`):
                the graph name
                
        Returns:
            True or False
        """
        return self.graph_engine.exec_rpp_fw(name, mode)
    
    @check_return
    def _graph_poll(self, name):
        r"""
        Poll sub graph model on deivece, wait interrupt.
        
        Args:
            name(`str`): 
                The graph name
                
        Returns: 
            True or False
        """
        return self.graph_engine.poll_rpp_fw(name)          
