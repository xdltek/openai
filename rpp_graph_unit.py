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
import argparse

from tqdm import tqdm
from loguru import logger
from collections import OrderedDict
from typing import List, Tuple, Dict
from functools import wraps

warnings.filterwarnings("ignore")
if platform.system().lower() == 'windows':
    sys.path.append('../lib/')
else:
    sys.path.append('/usr/local/rpp/lib')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pyfwgraphs as fw


class GraphUnit(object):
    def __init__(self, 
                 rpp_dir: str,
                 graph_name: str):
        r"""
        The RPP LLM base model implementation,

        Args:
            rpp_dir (`str`):
                The path to RPP ONNX models
        """
        super(GraphUnit, self).__init__()
        self.rpp_dir = rpp_dir
        self.graph_name = graph_name

        self._init_logger()
        self._init()
        
    def _init_logger(self, log_lvl: str='INFO'):
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
            obj (`List[int]`):
                The list contains the dimensions of the target tensor
        
        Returns:
            vol (`int`):
                The total volume(number of the elements) of the target tensor
        """
        vol = 1
        for elem in obj:
            vol *= elem
        return vol
    
    def _check_mse(x, y):
        error = np.power(np.abs(x.ravel() - y.ravel()), 2).sum()
        base = np.power(y.ravel(), 2).sum()
        mse = error / base
        return mse, error
    
    def _build_rpp_models(self):
        r"""
        Build the LLM prefill & decode engines on RPP.
        """
        self.graph_engine = fw.RppFwGraphs(self.rpp_dir)
        load_graph_start = time.perf_counter()
        logger.info('loading graph models...............')
        self.graph_engine.gen_rpp_fws()
        load_graph_end = time.perf_counter()
        load_graph_cost = load_graph_end - load_graph_start
        logger.info(f"load all graph time = {load_graph_cost:.2f} seconds.")
        self._allocate_buffers()
        
    def _alloc_graph_io_params(self, graph_name: str, ntype: int, mtype: int = 8):
        r"""
        Get the io informations for graph model

        Args:
            graph_name (`str`):
                the name of sub-graph
            ntype (`int`)
                1: inputs, 2: outputs, 3: inputs&outputs
            mtype (`int`)
                Memory allocation method: 
                IO_DATA_MODE_NULL = 1 
                IO_DATA_MODE_SHMGET = 2 
                IO_DATA_MODE_MALLOC = 4
                IO_DATA_MODE_PHYSIC = 8 
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

    def _init(self):
        r"""
        Init the models for RPP running
        """ 
        self._build_cpu_models()
        self._build_rpp_models()
        
    def _allocate_buffers(self):
        r"""
        Allocate the host buffers for graph
        """
        self.input_bindings, self.output_bindings = self._alloc_graph_io_params(self.graph_name,3)
        
    def _build_cpu_models(self):
        r"""
        Build the other models which is running in cpu.
        """
        
    def rpp_inference(self, inputs: Dict, outputs: Dict) -> Dict:
        r"""
        Inference the prompt with LLM models running on RPP.

        Args:
            kwargs (`Dict`):
                The input prompt from user
        """
        # copy inputs
        for key, value in inputs.items():
            if key in self.input_bindings:
                self.graph_engine.copy_data_to_device(value, value.nbytes, self.input_bindings[key].tensor_daddr)
            else:
                logger.error(f"inputs name is error: {key}")
                return None
        self.graph_engine.exec_rpp_fw_sync(self.graph_name)
        
        # copy outputs
        for key, value in outputs.items():
            if key in self.output_bindings:
                self.graph_engine.save_data_to_host(value, value.nbytes, self.output_bindings[key].tensor_daddr)
            else:
                logger.error(f"outputs name is error: {key}")
        return outputs
    
    def delete_rpp_fws(self):
        if self.graph_engine:
            del self.graph_engine
            self.graph_engine = None
            logger.info('delete rpp fws instance.............')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run graph unit test',
    )
    parser.add_argument('-g', '--graph_path', required=False, type=str, default="./graph_bins/")
    parser.add_argument('-n', '--graph_name', required=False, type=str, default="decode")
    args = parser.parse_args()
    
    # inputs data
    hidden_in = np.random.rand(1,1,3584).astype(np.float32)
    attention_mask = np.random.rand(1,1,1,8192).astype(np.float32)
    unsqueeze_output_0 = np.random.rand(1,1,1,128).astype(np.float32) # "/self_attn/Unsqueeze_output_0"
    unsqueeze_1_output_0 = np.random.rand(1,1,1,128).astype(np.float32) # '/self_attn/Unsqueeze_1_output_0'
    past_key = np.random.rand(1,8,8191,128).astype(np.float32)
    past_value = np.random.rand(1,8,8191,128).astype(np.float32)
    
    inputs = {}
    inputs['hidden_in'] = hidden_in
    inputs['attention_mask'] = attention_mask
    inputs['/self_attn/Unsqueeze_output_0'] = unsqueeze_output_0
    inputs['/self_attn/Unsqueeze_1_output_0'] = unsqueeze_1_output_0
    inputs['past_key_in0'] = past_key
    inputs['past_value_in0'] = past_value
    
    # outputs buffer, to save out data
    hidden_out = np.zeros((1,1,3584), dtype=np.float32)
    past_key0 = np.zeros((1,8,1,128), dtype=np.float32)
    past_value0 = np.zeros((1,8,1,128), dtype=np.float32)
    
    outputs = {}
    outputs["hidden_out"] = hidden_out
    outputs["key_quant_out"] = past_key0
    outputs["value_quant_out"] = past_value0
    
    unit_test = GraphUnit(rpp_dir=args.graph_path,
                          graph_name=args.graph_name)
    outputs = unit_test.rpp_inference(inputs, outputs)
    print(outputs)
    unit_test.delete_rpp_fws()