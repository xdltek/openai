import argparse
import platform
import time
import random
import struct
import os
from models import Qwen2Model, Llama3Model, Qwen2p5VLModel, Qwen2p5Model, Qwen3Model

global global_llm
global_llm = None

LLM_MODEL_NONE              = 0
LLM_QWEN1_7B_2K             = 1
LLM_QWEN2_7B_8K             = 10
LLM_QWEN2_7B_8K_STEP        = 11
LLM_QWEN2_7B_8K_STEP_KVBF   = 12
LLM_QWEN2_7B_8K_STEP_KVBF_NOCOPY   = 13
LLM_QWEN2P5_VL_7B_2k        = 20
LLM_QWEN2P5_7B_8k           = 25
LLM_QWEN3_8B_8k_STEP_NOCOPY = 30
LLM_LLAMA3_8B_4K            = 100
LLM_LLAMA3_8B_8K            = 101
LLM_LLAMA3_8B_8K_STEP       = 102
LLM_LLAMA3_8B_8K_STEP_KVNOCOPY     = 103
LLM_DEEPSEEK_QWEN2_8B_8K           = 200
LLM_DEEPSEEK_LLAMA3_8B_8K          = 210

MODEL_INFO_FORMAT = 'q i 128s i i i q 64s 8s 16s 64s 16s i 256s'

class ModelInfo:
    def __init__(self):
        self.nflag = 0  # model identification
        self.nsize = 0  # size of model information
        self.head_reserved = b""  # graph reserve
        self.input_size = 0  # maximum input size
        self.output_size = 0  # maximum output size
        self.total_size = 0  # maximum total size
        self.created_timestamp = 0  # create timestamp
        self.original_model_name = b""  # original model name
        self.parameter_quantity = b""  # parameter quantity
        self.quantization_method = b""  # quantization method
        self.xdl_model_name = b""  # xdl model name
        self.xdl_model_version = b""  # xdl model version
        self.xdl_model_type = 0  # xdl model version
        self.reserved = b""  # usr reserve

    @classmethod
    def pack(self):
        return struct.pack(
            MODEL_INFO_FORMAT,
            self.nflag,
            self.nsize,
            self.head_reserved,
            self.input_size,
            self.output_size,
            self.total_size,
            self.created_timestamp,
            self.original_model_name,
            self.parameter_quantity,
            self.quantization_method,
            self.xdl_model_name,
            self.xdl_model_version,
            self.xdl_model_type,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data):
        unpacked_data = struct.unpack(MODEL_INFO_FORMAT, data)
        model_info = cls()
        model_info.nflag = unpacked_data[0]
        model_info.nsize = unpacked_data[1]
        model_info.head_reserved = unpacked_data[2].rstrip(b'\x00')
        model_info.input_size = unpacked_data[3]
        model_info.output_size = unpacked_data[4]
        model_info.total_size = unpacked_data[5]
        model_info.created_timestamp = unpacked_data[6]
        model_info.original_model_name = unpacked_data[7].rstrip(b'\x00')
        model_info.parameter_quantity = unpacked_data[8].rstrip(b'\x00')
        model_info.quantization_method = unpacked_data[9].rstrip(b'\x00')
        model_info.xdl_model_name = unpacked_data[10].rstrip(b'\x00')
        model_info.xdl_model_version = unpacked_data[11].rstrip(b'\x00')
        model_info.xdl_model_type = unpacked_data[12]
        model_info.reserved = unpacked_data[13].rstrip(b'\x00')
        return model_info

    def __str__(self):
        return (f"ModelInfo(nflag={self.nflag}, nsize={self.nsize}, "
                f"head_reserved={self.head_reserved}, "
                f"input_size={self.input_size}, "
                f"output_size={self.output_size}, total_size={self.total_size}, "
                f"created_timestamp={self.created_timestamp}, "
                f"original_model_name={self.original_model_name.decode()}, "
                f"parameter_quantity={self.parameter_quantity.decode()}, "
                f"quantization_method={self.quantization_method.decode()}, "
                f"xdl_model_name={self.xdl_model_name.decode()}, "
                f"xdl_model_version={self.xdl_model_version.decode()}, "
                f"xdl_model_type={self.xdl_model_type}, "
                f"reserved={self.reserved.decode()}")

def load_model_info(file_path):
    if os.path.exists(os.path.join(file_path, './g_version.bin')):
        g_version_file = os.path.join(file_path, 'g_version.bin')
    else:
        g_version_file = os.path.join(file_path, 'firmware.pb')
    with open(g_version_file, 'rb') as f:
        packed_data = f.read(struct.calcsize(MODEL_INFO_FORMAT))
        
    model_info = ModelInfo.unpack(packed_data)
    print("Loaded ModelInfo:")
    print(model_info)
    print("")
    return model_info

def get_prompt(run_mode, prompt_file):
    if '.json' in prompt_file:
        do_infer_text = input("Do you want to do inference? [Y]yes, [N]no: ")
        if do_infer_text == "y" or do_infer_text == "Y":
            return prompt_file
        elif do_infer_text == "n" or do_infer_text == "N":
            release_resource()
    if run_mode == 0:
        prompt = input("Please input you prompt:")
    elif run_mode == 1:
        with open(prompt_file, 'r',  encoding="utf-8") as f:
            prompts = f.readlines()
        prompt = random.choice(prompts)
    else:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    return prompt

def release_resource():
    print("exiting the program...")
    if global_llm:
        global_llm.stop_inference(True)
        time.sleep(1)
        global_llm.delete_rpp_fws()
        exit(1)

def register_exit_handler():
    if platform.system().lower() == 'windows':
        import win32api
        # 当用户关闭Console时，系统会发送次消息
        win32api.SetConsoleCtrlHandler(lambda signum: signum == 2 and release_resource(), True)
        win32api.SetConsoleCtrlHandler(lambda signum: signum == 0 and release_resource(), True)
        win32api.SetConsoleCtrlHandler(lambda signum: signum == 1 and release_resource(), True)
        win32api.SetConsoleCtrlHandler(lambda signum: signum == 5 and release_resource(), True)
        win32api.SetConsoleCtrlHandler(lambda signum: signum == 6 and release_resource(), True)
    # other platform
    else:
        import signal
        signal.signal(signal.SIGHUP,  lambda signum, frame: release_resource())
        signal.signal(signal.SIGINT,  lambda signum, frame: release_resource())
        signal.signal(signal.SIGTSTP, lambda signum, frame: release_resource())
        signal.signal(signal.SIGTERM, lambda signum, frame: release_resource())
        signal.signal(signal.SIGQUIT, lambda signum, frame: release_resource())

def main():
    parser = argparse.ArgumentParser(
        description='run qwen graph demo',
    )
    parser.add_argument('-g', '--graph_path', required=False, type=str, default="./graph_bins")
    parser.add_argument('-w', '--write_file', required=False, type=int, default=0)
    parser.add_argument('-l', '--low_power', required=False, type=int, default=0)
    parser.add_argument('-i', '--input_size', required=False, type=int, default=8192)
    parser.add_argument('-t', '--target_len', required=False, type=int, default=8192)
    parser.add_argument('-f', '--prefix', required=False, type=int, default=1)
    parser.add_argument('-d', '--do_sample', required=False, type=int, default=1)
    parser.add_argument('-p', '--perf_mode', required=False, type=int, default=0)
    parser.add_argument('-r', '--run_mode', required=False, type=int, default=0)
    parser.add_argument('-pf', '--prompt_file', required=False, type=str, default='./prompts/prompt.txt')
    args = parser.parse_args()
    
    SUPPORT_TYPE = {
        LLM_QWEN2_7B_8K_STEP_KVBF_NOCOPY: Qwen2Model,
        LLM_LLAMA3_8B_8K_STEP_KVNOCOPY: Llama3Model,
        LLM_QWEN2P5_VL_7B_2k: Qwen2p5VLModel,
        LLM_QWEN2P5_7B_8k: Qwen2p5Model,
        LLM_QWEN3_8B_8k_STEP_NOCOPY: Qwen3Model
    }
    
    model_info = load_model_info(args.graph_path)
    cls_func = SUPPORT_TYPE.get(model_info.xdl_model_type, None)
    if cls_func is None:
        raise RuntimeError(f'\nNot support model type: {model_info.xdl_model_type}\nSupport model list: {list(SUPPORT_TYPE.keys())}, please check')

    kwargs = {
        'rpp_dir': args.graph_path,
        'input_size': args.input_size,
        'target_len': args.target_len,
        'write_file': args.write_file,
        'low_power': args.low_power,
        'prefix': args.prefix,
        'perf_mode':args.perf_mode
    }
    global global_llm
    global_llm = cls_func(**kwargs)
    
    global_llm.set_infer_params(penalty=1.1, 
                                top_k=40, 
                                top_p=0.9, 
                                temperature=0.2, 
                                min_tokens_to_keep=1,
                                do_sample=args.do_sample)
    # register signal
    register_exit_handler()

    while True:
        prompt = get_prompt(args.run_mode, args.prompt_file)
        if len(prompt) == 0:
            print("your prompt is None, please input prompts.")
        else:
            global_llm.rpp_inference(prompt=prompt, tools=None)

if __name__ == '__main__':
    main()
    