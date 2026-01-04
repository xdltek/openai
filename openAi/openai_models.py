import argparse
import time
import struct
import os
import pprint
import asyncio
import uuid
import base64
import tempfile
import json
import sys
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from models import Qwen2p5Model, Qwen3Model, Qwen2p5VLModel
from commons import LlmBaseModel
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse
from .openai_common import MessageParse,ChatMessage,ChatCompletionRequest,ChatCompletionResponse,ModelInfo,ChatCompletionResponseChoice 
from .openai_common import ChatCompletionResponseStreamChoice, DeltaMessage
from fastapi import FastAPI, Request
from urllib.request import urlopen
from io import BytesIO
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


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
LLM_DEEPSEEK_QWEN2_8B_8K    = 200
LLM_DEEPSEEK_LLAMA3_8B_8K   = 210

def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:
        return data.json(*args, **kwargs)

class OpenAiModel(object):
    def __init__(self, **kwargs):
        SUPPORT_TYPE = {
            LLM_QWEN2P5_7B_8k: Qwen2p5Model,
            LLM_QWEN3_8B_8k_STEP_NOCOPY: Qwen3Model,
            LLM_QWEN2P5_VL_7B_2k: Qwen2p5VLModel
        }
        self.rpp_dir = kwargs.get('rpp_dir')
        self.do_sample = kwargs.get('do_sample')
        kwargs.pop("do_sample", None)
        model_info: ModelInfo = self._load_model_info(self.rpp_dir)
        cls_func = SUPPORT_TYPE.get(model_info.xdl_model_type, None)
        if cls_func is None:
            raise RuntimeError(f'\nNot support model type: {model_info.xdl_model_type}\nSupport model list: {list(SUPPORT_TYPE.keys())}, please check')
        kwargs['input_size'] = model_info.input_size
        kwargs['target_len'] = model_info.total_size
        self.llm_model: LlmBaseModel = cls_func(**kwargs)
        self.llm_model.set_infer_params(penalty=1.05, 
                                        top_k=40, 
                                        top_p=0.9, 
                                        temperature=0.2, 
                                        min_tokens_to_keep=1,
                                        do_sample=self.do_sample)
        self.lock = asyncio.Lock()
        self.is_vision_model = (model_info.xdl_model_type == LLM_QWEN2P5_VL_7B_2k)
        self._temp_files = []  # Track temporary files for cleanup
        
    def _load_model_info(self,
                         file_path) -> ModelInfo:
        MODEL_INFO_FORMAT = 'q i 128s i i i q 64s 8s 16s 64s 16s i 256s'
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
    
    def _convert_openai_messages_to_qwen_format(self, messages: List[ChatMessage]) -> List[Dict]:
        """
        Convert OpenAI API messages format to Qwen format for vision models.
        
        OpenAI format:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..." or "https://..."}}
            ]
        }
        
        Qwen format:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image", "image": "path_or_url"}
            ]
        }
        """
        qwen_messages = []
        temp_files = []  # Keep track of temp files for cleanup
        
        for msg in messages:
            qwen_msg = {"role": msg.role}
            
            # Handle None content
            if msg.content is None:
                qwen_msg["content"] = ""
            elif isinstance(msg.content, str):
                # Simple text message
                qwen_msg["content"] = msg.content
            elif isinstance(msg.content, list):
                # Multi-modal message
                qwen_content = []
                for item in msg.content:
                    # Handle both dict and BaseModel instances
                    if isinstance(item, dict):
                        item_dict = item
                    elif hasattr(item, 'dict'):
                        item_dict = item.dict()
                    elif hasattr(item, 'model_dump'):
                        item_dict = item.model_dump()
                    else:
                        # Fallback: treat as text
                        qwen_content.append({"type": "text", "text": str(item)})
                        continue
                    
                    item_type = item_dict.get("type")
                    if item_type == "text":
                        qwen_content.append({"type": "text", "text": item_dict.get("text", "")})
                    elif item_type == "image_url":
                        # Handle OpenAI image_url format
                        image_url = item_dict.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = image_url
                        
                        # Handle base64 images
                        if url.startswith("data:image"):
                            # Extract base64 data - MUST convert to file, process_vision_info doesn't handle data URLs well
                            if HAS_PIL:
                                try:
                                    header, data = url.split(",", 1)
                                    # Fix base64 padding if needed
                                    missing_padding = len(data) % 4
                                    if missing_padding:
                                        data += '=' * (4 - missing_padding)
                                    image_data = base64.b64decode(data, validate=True)
                                    # Save to temporary file
                                    img = Image.open(BytesIO(image_data))
                                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                                    img.save(temp_file.name, format="JPEG")
                                    temp_file.close()  # Close the file so it can be read by other processes
                                    temp_files.append(temp_file.name)
                                    qwen_content.append({"type": "image", "image": temp_file.name})
                                except Exception as e:
                                    # If conversion fails, raise error - we can't proceed without a valid image file
                                    print(f"Error: Could not convert base64 image to file: {e}")
                                    raise ValueError(f"Failed to process base64 image: {e}")
                            else:
                                # PIL not available - we must have PIL for base64 images
                                raise ValueError("PIL (Pillow) is required to process base64 images. Please install it: pip install Pillow")
                        elif url.startswith("http://") or url.startswith("https://"):
                            # URL - Qwen can handle URLs directly
                            qwen_content.append({"type": "image", "image": url})
                        else:
                            # Assume it's a file path
                            qwen_content.append({"type": "image", "image": url})
                    elif item_type == "image":
                        # Already in Qwen format
                        qwen_content.append(item_dict)
                    else:
                        # Fallback: treat as text
                        qwen_content.append({"type": "text", "text": str(item_dict)})
                
                qwen_msg["content"] = qwen_content
            else:
                # Fallback: convert to string
                qwen_msg["content"] = str(msg.content) if msg.content else ""
            
            qwen_messages.append(qwen_msg)
        
        # Store temp files for cleanup later
        self._temp_files.extend(temp_files)
        
        
        return qwen_messages
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created for base64 image conversion."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_file}: {e}")
        self._temp_files.clear()
    
    def _chat_full(self,
                   query: str,
                   history: List[List[str]], 
                   stop_words: List[str],
                   gen_kwargs: Dict,
                   system: str,
                   messages: Union[List[List[str]], List[ChatMessage]], 
                   tools: List[List[str]],
                   api_request: Request): 
        try:
            if self.is_vision_model:
                # For vision models, convert OpenAI messages to Qwen format
                if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], ChatMessage):
                    try:
                        qwen_messages = self._convert_openai_messages_to_qwen_format(messages)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise
                else:
                    # Already in Qwen format or file path
                    qwen_messages = messages
                
                self.llm_model.rpp_inference(prompt=qwen_messages)
            else:
                # For non-vision models, use tokenizer-based approach
                stop_words_ids = [self.llm_model.tokenizer.encode(s) 
                                  for s in stop_words] if stop_words else None
                self.llm_model.rpp_inference(prompt=messages, 
                                             tools=tools, 
                                             stop_words_ids=stop_words_ids)
        finally:
            # Clean up temporary files after inference
            if self.is_vision_model:
                self._cleanup_temp_files()
        
        # Get response text - ensure it's a string
        response_text = getattr(self.llm_model, 'str_tokens', '') or ""
        if response_text and not isinstance(response_text, str):
            response_text = str(response_text)
        
        return {"response": response_text,
                "prompt_tokens": getattr(self.llm_model, 'prompt_tokens', 0),
                "completion_tokens": getattr(self.llm_model, 'completion_tokens', 0),
                "total_tokens": getattr(self.llm_model, 'total_tokens', 0)}
    
    async def _chat_stream(self,
                           query: str,
                           history: List[List[str]],
                           model_id: str,
                           stop_words: List[str],
                           gen_kwargs: Dict,
                           system: str,
                           messages: Union[List[List[str]], List[ChatMessage]],
                           tools: List[List[str]],
                           api_request: Request):
        async with self.lock:
            try:
                # Initial chunk
                choice_data = ChatCompletionResponseStreamChoice(index=0, 
                                                                 delta=DeltaMessage(role='assistant'), 
                                                                 finish_reason=None)
                chunk = ChatCompletionResponse(id="",
                                               object='chat.completion',
                                               model=model_id,
                                               choices=[choice_data],
                                               usage={"prompt_tokens": 0,
                                                      "completion_tokens": 0,
                                                      "total_tokens": 0})
                yield _dump_json(chunk, exclude_unset=True)
                
                # Prepare messages for model
                if self.is_vision_model:
                    # For vision models, convert OpenAI messages to Qwen format
                    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], ChatMessage):
                        qwen_messages = self._convert_openai_messages_to_qwen_format(messages)
                    else:
                        qwen_messages = messages
                    response_generator = self.llm_model.rpp_inference_stream(prompt=qwen_messages)
                else:
                    stop_words_ids = [self.llm_model.tokenizer.encode(s) 
                                      for s in stop_words] if stop_words else None
                    response_generator = self.llm_model.rpp_inference_stream(prompt=messages, tools=tools, stop_words_ids=stop_words_ids)

                # Stream tokens from model
                async for token_text in response_generator:
                    # Check if client disconnected
                    if await api_request.is_disconnected():
                        print(f"\nclient is disconnected, addr: {api_request.client}\n", file=sys.stderr)
                        self.stop_inference(True)
                        break
                    
                    # Skip None or empty tokens
                    if not token_text:
                        continue
                    
                    # Create chunk for this token
                    choice_data = ChatCompletionResponseStreamChoice(index=0, 
                                                                     delta=DeltaMessage(role='assistant', content=token_text), 
                                                                     finish_reason=None)
                    chunk = ChatCompletionResponse(id="cmpl-467f59d348784a0a8c2a325d022367bb",
                                                   object='chat.completion.chunk',
                                                   model=model_id,
                                                   choices=[choice_data],
                                                   usage={"prompt_tokens": 0,
                                                          "completion_tokens": 0,
                                                          "total_tokens": 0})
                    yield _dump_json(chunk, exclude_unset=True)
                
            except Exception as e:
                import traceback
                print(f"Error in streaming: {e}", file=sys.stderr)
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            finally:
                # Always send final chunk with finish_reason
                try:
                    prompt_tokens = getattr(self.llm_model, 'prompt_tokens', 0)
                    completion_tokens = getattr(self.llm_model, 'completion_tokens', 0)
                    total_tokens = getattr(self.llm_model, 'total_tokens', 0)
                except:
                    prompt_tokens = completion_tokens = total_tokens = 0
                
                choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                                 delta=DeltaMessage(),
                                                                 finish_reason='stop')
                chunk = ChatCompletionResponse(id="cmpl-467f59d348784a0a8c2a325d022367bb",
                                               object='chat.completion.chunk',
                                               model=model_id,
                                               choices=[choice_data],
                                               usage={"prompt_tokens": prompt_tokens,
                                                      "completion_tokens": completion_tokens,
                                                      "total_tokens": total_tokens})
                yield _dump_json(chunk, exclude_unset=True)
                yield '[DONE]'
        
    def chat_completion(self,
                        request: ChatCompletionRequest,
                        api_request: Request):
        gen_kwargs = {}
        if request.top_k is not None:
            gen_kwargs['top_k'] = request.top_k
        if request.temperature is not None:
            if request.temperature < 0.01:
                gen_kwargs['top_k'] = 1  # greedy decoding
            else:
                # Not recommended. Please tune top_p instead.
                gen_kwargs['temperature'] = request.temperature
        if request.top_p is not None:
            gen_kwargs['top_p'] = request.top_p
            
        self.llm_model.set_infer_params(**gen_kwargs)
        
        stop_words = MessageParse.add_extra_stop_words(request.stop)
        if request.tools:
            stop_words = stop_words or []
            if 'Observation:' not in stop_words:
                stop_words.append('Observation:')

        # query, history, system = MessageParse.parse_messages(request.messages, 
        #
        query = None
        history = None 
        system = None
          
        if request.stream:
            # if request.tools:
            #     raise HTTPException(status_code=400,
            #                         detail='Invalid request: Function calling is not yet implemented for stream mode.')
            generate = self._chat_stream(query=query,
                                         history=history,
                                         model_id=request.model,
                                         stop_words=stop_words,
                                         gen_kwargs=gen_kwargs,
                                         system=system,
                                         messages=request.messages,
                                         tools=request.tools,
                                         api_request=api_request)
            return EventSourceResponse(generate, media_type='text/event-stream')

        # if query is object():
        #     stop_words_ids = [self.llm_model.tokenizer.encode(s) 
        #                       for s in stop_words] if stop_words else None
        #     response = text_complete_last_message(history,
        #                                         stop_words_ids=stop_words_ids,
        #                                         gen_kwargs=gen_kwargs,
        #                                         system=system)
        else:
            res_dict = self._chat_full(query=query,
                                       history=history,
                                       system=system,
                                       gen_kwargs=gen_kwargs,
                                       messages=request.messages,
                                       tools=request.tools,
                                       stop_words=stop_words,
                                       api_request=api_request)
            
            # Extract response - ensure it's a string
            response = res_dict.get('response', '')
            if response is None:
                response = ""
            elif not isinstance(response, str):
                response = str(response)
            
            # Fallback: if response is empty, try to get from llm_model directly
            if not response or len(response.strip()) == 0:
                response = getattr(self.llm_model, 'str_tokens', '') or ""
        
        # Ensure response is a string, not None
        if response is None:
            response = ""
        elif not isinstance(response, str):
            response = str(response)
        
        response = MessageParse.trim_stop_words(response, stop_words)
        
        # Ensure response is not None
        if response is None:
            response = ""
        
        choice_data = ChatCompletionResponseChoice(index=0,
                                                   message=ChatMessage(role='assistant', content=response),
                                                   finish_reason='stop')
        
        result = ChatCompletionResponse(id=f"chatcmpl-tool-{str(uuid.uuid4()).replace('-', '')[:24]}",
                                      object='chat.completion',
                                      model=request.model,
                                      choices=[choice_data],
                                      usage={
                                      "prompt_tokens": res_dict.get('prompt_tokens', 0),
                                      "completion_tokens": res_dict.get('completion_tokens', 0),
                                      "total_tokens": res_dict.get('total_tokens', 0)})
        
        
        return result
        
    def stop_inference(self, 
                       bstop: bool=False):
        self.llm_model.stop_infer = bstop