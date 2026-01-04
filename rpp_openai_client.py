from openai import OpenAI
import requests
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

data = {
      "model": "qwen2.5",
      "stream": False,
      "messages":[
        {
            "role": "system",
            "content": """你是一个很有帮助的助手。如果用户提问关于天气的问题，请调用'get_current_weather'函数;
            如果用户提问关于时间的问题，请调用'get_current_time'函数。
            请以友好的语气回答问题。""",
        },
        {
            "role": "user",
            "content": "西安天气"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "当你想知道现在的时间时非常有用。",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "当你想查询指定城市的天气时非常有用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
}

messages = [
    {
        "role": "system",
        "content": """你是一个很有帮助的助手。如果用户提问关于天气的问题，请调用'get_current_weather'函数;
         如果用户提问关于时间的问题，请调用'get_current_time'函数。
         请以友好的语气回答问题。""",
    },
    {
        "role": "user",
        #"content": "西安当前时间和天气"
        "content": "你是谁，简单介绍下自己"
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    }
                },
                "required": ["location"]
            }
        }
    }
]

for chunk in client.chat.completions.create(model="Qwen2.5",
                                            messages=messages,
                                            tools=tools,
                                            stream=True
                                            ):
    if hasattr(chunk.choices[0].delta, "content"):
        if (chunk.choices[0].delta.content == None):
            continue
        print(chunk.choices[0].delta.content, end="", flush=True)
print("")

# chunk = requests.post('http://127.0.0.1:8000/v1/chat/completions', json=data)
# print(chunk.json())

# chunk = client.chat.completions.create(model="Qwen2.5",
#                                        messages=messages,
#                                        tools=tools,
#                                        stream=False)
# print("")
# # if hasattr(chunk.choices[0].message, "content"):
# #     if (chunk.choices[0].message.content == None):
# #         print("message is none")
# #     print(chunk.choices[0].message.content, end="", flush=True)

# print(chunk.choices[0].message.model_dump(),"")
# print("")
