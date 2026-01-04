
import time
import time
import os
import warnings
import base64
import copy
import json
import time
import struct
import re
import uuid

from typing import Dict, List, Literal, Optional, Union, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from pydantic import BaseModel
from typing import List
from urllib.request import urlopen
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_INFO_FORMAT = 'q i 128s i i i q 64s 8s 16s 64s 16s i 256s'

TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
    ' What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}'
)

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(
            f'{username}:{password}'.encode()).decode()
        
    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization:
            try:
                schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass

        headers = {'WWW-Authenticate': 'Basic'}
        return Response(status_code=401, headers=headers)

class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[Any] = None  # Can be string or list of content blocks (for vision models)
    tool_calls: Optional[List[Dict]] = None
    function_call: Optional[Dict] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Union[ChatMessage]
    finish_reason: Literal['stop', 'length', 'function_call', 'tool_calls']

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]

class UsageData(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int    

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    usage: UsageData

class Words(BaseModel):
    word: str
    start: float
    end: float
    probability: float

class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    token: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: List[Words]

class TranscriptionResponse(BaseModel):
    text: str
    segments: List[Segment]
    language: str

class MessageParse(object):
    @staticmethod
    def add_extra_stop_words(stop_words):
        if stop_words:
            _stop_words = []
            _stop_words.extend(stop_words)
            for x in stop_words:
                s = x.lstrip('\n')
                if s and (s not in _stop_words):
                    _stop_words.append(s)
            return _stop_words
        return stop_words

    @staticmethod
    def trim_stop_words(response, stop_words):
        if stop_words:
            for stop in stop_words:
                idx = response.find(stop)
                if idx != -1:
                    response = response[:idx]
        return response

    @staticmethod
    def parse_messages(messages, functions):
        if all(m.role != 'user' for m in messages):
            raise HTTPException(
                status_code=400,
                detail='Invalid request: Expecting at least one user message.',
            )

        messages = copy.deepcopy(messages)
        if messages[0].role == 'system':
            system = messages.pop(0).content.lstrip('\n').rstrip()
        else:
            system = 'You are a helpful assistant.'

        if functions:
            tools_text = []
            tools_name_text = []
            for func_info in functions:
                func_info = func_info.get('function')
                name = func_info.get('name', '')
                name_m = func_info.get('name_for_model', name)
                name_h = func_info.get('name_for_human', name)
                desc = func_info.get('description', '')
                desc_m = func_info.get('description_for_model', desc)
                tool = TOOL_DESC.format(
                    name_for_model=name_m,
                    name_for_human=name_h,
                    # Hint: You can add the following format requirements in description:
                    #   "Format the arguments as a JSON object."
                    #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                    description_for_model=desc_m,
                    parameters=json.dumps(func_info['parameters'],ensure_ascii=False) if 'parameters' in func_info else ''
                )
                tools_text.append(tool)
                tools_name_text.append(name_m)
            tools_text = '\n\n'.join(tools_text)
            tools_name_text = ', '.join(tools_name_text)
            instruction = (REACT_INSTRUCTION.format(
                tools_text=tools_text,
                tools_name_text=tools_name_text,
            ).lstrip('\n').rstrip())
        else:
            instruction = ''

        messages_with_fncall = messages
        messages = []
        for m_idx, m in enumerate(messages_with_fncall):
            role, content, func_call = m.role, m.content, m.function_call
            content = content or ''
            content = content.lstrip('\n').rstrip()
            if role == 'function':
                if (len(messages) == 0) or (messages[-1].role != 'assistant'):
                    raise HTTPException(
                        status_code=400,
                        detail=
                        'Invalid request: Expecting role assistant before role function.',
                    )
                messages[-1].content += f'\nObservation: {content}'
                if m_idx == len(messages_with_fncall) - 1:
                    # add a prefix for text completion
                    messages[-1].content += '\nThought:'
            elif role == 'assistant':
                if len(messages) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=
                        'Invalid request: Expecting role user before role assistant.',
                    )
                if func_call is None:
                    if functions:
                        content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
                else:
                    f_name, f_args = func_call['name'], func_call['arguments']
                    if not content.startswith('Thought:'):
                        content = f'Thought: {content}'
                    content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
                if messages[-1].role == 'user':
                    messages.append(
                        ChatMessage(role='assistant',
                                    content=content.lstrip('\n').rstrip()))
                else:
                    messages[-1].content += '\n' + content
            elif role == 'user':
                messages.append(
                    ChatMessage(role='user',
                                content=content.lstrip('\n').rstrip()))
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f'Invalid request: Incorrect role {role}.')

        query = _TEXT_COMPLETION_CMD
        if messages[-1].role == 'user':
            query = messages[-1].content
            messages = messages[:-1]

        if len(messages) % 2 != 0:
            raise HTTPException(status_code=400, detail='Invalid request')

        history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
        for i in range(0, len(messages), 2):
            if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
                usr_msg = messages[i].content.lstrip('\n').rstrip()
                bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
                if instruction and (i == len(messages) - 2):
                    usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                    instruction = ''
                history.append([usr_msg, bot_msg])
            else:
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting exactly one user (or function) role before every assistant role.',
                )
        if instruction:
            assert query is not _TEXT_COMPLETION_CMD
            query = f'{instruction}\n\nQuestion: {query}'
        return query, history, system

    @staticmethod
    def parse_response(response):
        ## for new tools_call
        tool_call_matches = re.findall(r'<tool_call>\n(.*?)\n</tool_call>', response, re.DOTALL)
        if tool_call_matches:
            content = response.split('<tool_call>')[0].strip()
            content = content if content else None
            tool_calls = []
            for tool_call_json in tool_call_matches:
                try:
                    tool_data = json.loads(tool_call_json)
                    call_id = f"chatcmpl-tool-{str(uuid.uuid4()).replace('-', '')[:24]}"
                    tool_call = {'id': call_id,
                                 'function': {'name': tool_data.get('name', ''),
                                              'arguments': json.dumps(tool_data.get('arguments', {}),ensure_ascii=False),
                                              # 'arguments': tool_data.get('arguments', {}),
                                              },
                                 'type':'function'
                                 }
            
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue
            
            choice_data = ChatCompletionResponseChoice(index=0,
                                                       message=ChatMessage(role='assistant',
                                                                           content=content,
                                                                           tool_calls=tool_calls,
                                                                           function_call=None),
                                                       finish_reason='tool_calls',)
            return choice_data
        
        ## for old function_call
        func_name, func_args = '', ''
        i = response.find('\nAction:')
        j = response.find('\nAction Input:')
        k = response.find('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is omitted by the LLM,
                # because the output text may have discarded the stop word.
                response = response.rstrip() + '\nObservation:'  # Add it back.
            k = response.find('\nObservation:')
            func_name = response[i + len('\nAction:'):j].strip()
            func_args = response[j + len('\nAction Input:'):k].strip()
        if func_name:
            response = response[:i]
            t = response.find('Thought: ')
            if t >= 0:
                response = response[t + len('Thought: '):]
            response = response.strip()
            choice_data = ChatCompletionResponseChoice(index=0,
                                                       message=ChatMessage(role='assistant',
                                                                           content=response,
                                                                           function_call={'name': func_name,
                                                                                          'arguments': func_args},),
                                                        finish_reason='function_call',
            )
            return choice_data

        z = response.rfind('\nFinal Answer: ')
        if z >= 0:
            response = response[z + len('\nFinal Answer: '):]
        choice_data = ChatCompletionResponseChoice(index=0,
                                                   message=ChatMessage(role='assistant', content=response),
                                                   finish_reason='stop',)
        return choice_data
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
        
        


#### for test
if __name__ == '__main__':
    msg = [
    {
      "role": "system",
      "content": "## 您是顶级个人助理;\n## 默认用简体中文回答用户问题.\n..."
    },
    {
      "role": "user",
      "content": "你真棒"
    },
    {
      "role": "assistant",
      "content": "谢谢你的夸奖"
    },
    {
      "role": "user",
      "content": "根据文件列表回答：[{\\\"path\\\":\\\"source7/\\\",\\\"name\\\":\\\"我的文档\\\",\\\"type\\\":\\\"folder\\\",\\\"size\\\":21573,\\\"pathDisplay\\\":\\\"/个人空间/我的文档/\\\",\\\"parentLevel\\\":\\\",0,5,\\\",\\\"sourceID\\\":7,\\\"createTime\\\":\\\"2025-06-20 15:40\\\",\\\"modifyTime\\\":\\\"2025-06-20 17:43\\\",\\\"childHas\\\":{\\\"fileNum\\\":1,\\\"folderNum\\\":0},\\\"sizeShow\\\":\\\"21.1KB\\\",\\\"fid\\\":\\\"54-0\\\"}]\n当前目录名是什么？"
    }
  ]
    fun = [
        {
            "type": "function",
            "function": {
                "name": "getUrlContent",
                "description": "当需要获取url内容时调用本插件",
                "parameters": {
                    "properties": {
                        "links": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "url链接,最多10个."
                        }
                    },
                    "type": "object",
                    "required": [
                        "links"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "emailSend",
                "description": "将文本内容发送邮件(当用户提到发送邮件时调用;邮件内容根据用户提问进行处理;不支持附件,请勿提及附件)",
                "parameters": {
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "邮件标题,由用户提供或根据内容自动生成"
                        },
                        "emailTo": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "发送邮箱地址,最多10个."
                        },
                        "emailCC": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "抄送邮箱地址,最多10个."
                        },
                        "content": {
                            "type": "string",
                            "description": "邮件内容"
                        }
                    },
                    "type": "object",
                    "required": [
                        "title",
                        "emailTo",
                        "content"
                    ]
                }
            }
        }
    ]
    req = ChatCompletionRequest(model='qwen', messages=msg, functions=fun)
    MessageParse.parse_messages(messages=req.messages, functions=req.functions)