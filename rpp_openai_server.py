import argparse
import uvicorn
from openAi import OpenAiModel, ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openAi import Daemon
import os
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/v1/models', response_model=ModelList)
async def list_models():
    model_card = ModelCard(id='qwen2')
    return ModelList(data=[model_card])

@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest,
                                 api_request: Request):
    global openai_llm
    return openai_llm.chat_completion(request, api_request)

class OpenAiDaemon(Daemon):
    def run(self):
        if os.path.exists('/dev/shm/rpp_dev_shared_mem'):
            os.remove('/dev/shm/rpp_dev_shared_mem')
        host = self.kwargs.pop("host", None)
        port = self.kwargs.pop("port", None)
        global openai_llm
        openai_llm = OpenAiModel(**(self.kwargs))
        uvicorn.run(app, host=host, port=port, workers=1)
        
if __name__ == '__main__':
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
    parser.add_argument('-daemon', '--daemon_mode', required=False, type=int, default=1)
    parser.add_argument('--server-port',
                        type=int,
                        default=8001,
                        help='Demo server port.')
    parser.add_argument('--server-name',
                        type=str,
                        default='127.0.0.1',
                        help=
                        'Demo server name. Default: 127.0.0.1, which is only visible from the local computer.'
                        'If you want other computers to access your server, use 0.0.0.0 instead.',
    )
    args = parser.parse_args()
    
    kwargs = {
        'rpp_dir': args.graph_path,
        'input_size': args.input_size,
        'target_len': args.target_len,
        'write_file': args.write_file,
        'low_power': args.low_power,
        'prefix': args.prefix,
        'perf_mode':args.perf_mode,
        'do_sample':args.do_sample
    }
    
    if args.daemon_mode:
        kwargs['host'] = args.server_name
        kwargs['port'] = args.server_port
        PIDFILE = '/tmp/daemon-openai.pid'
        LOG = '/tmp/daemon-openai.log'
        daemon = OpenAiDaemon(pidfile=PIDFILE, stdout=LOG, stderr=LOG, **kwargs)
        daemon.start()
    else:
        if os.path.exists('/dev/shm/rpp_dev_shared_mem'):
            os.remove('/dev/shm/rpp_dev_shared_mem')
        global openai_llm
        openai_llm = OpenAiModel(**kwargs)
        uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)