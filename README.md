# RPP OpenAI-Compatible API Server

An OpenAI-compatible API server for running Large Language Models (LLMs) on RPP (**Reconfigurable Parallel Processor**). This server provides a standard OpenAI API interface for text generation, vision-language tasks, and function calling.

## Features

- **OpenAI-Compatible API**: Full compatibility with OpenAI's Chat Completions API
- **Multiple Model Support**: Qwen2.5, Qwen3, and Qwen2.5VL (Vision-Language)
- **Streaming Support**: Real-time token streaming via Server-Sent Events (SSE)
- **Vision-Language Models**: Support for image understanding and multi-modal tasks
- **Function Calling**: Support for tools and function calling
- **Daemon Mode**: Run as a background daemon process
- **Flexible Configuration**: Customizable inference parameters and server settings

## Supported Models

The server automatically detects the model type from the graph directory. Currently supported models:

- **Qwen2.5 7B 8K** (`LLM_QWEN2P5_7B_8k`)
- **Qwen3 8B 8K** (`LLM_QWEN3_8B_8k_STEP_NOCOPY`)
- **Qwen2.5VL 7B 2K** (`LLM_QWEN2P5_VL_7B_2k`) - Vision-Language model

## Installation

### Prerequisites

1. **Conda Environment**
   ```bash
   # Install Miniconda (Linux)
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # Create conda environment
   conda create -n xdl_openai python=3.11
   conda activate xdl_openai
   ```

2. **Install Dependencies**
   ```bash
   cd utils/run_graph_utils
   pip install -r requirements.txt
   ```

3. **RPP Libraries**
   The following libraries must be available:
   - `pyrt.cpython-311-x86_64-linux-gnu.so` (path: `/usr/local/rpp/lib/`)
   - `pyfwgraphs.cpython-311-x86_64-linux-gnu.so` (path: `/usr/local/rpp/lib/fw/`)
   
   Create symbolic links if needed:
   ```bash
   sudo ln -s /usr/local/rpp/lib/pyrt.cpython-311-x86_64-linux-gnu.so pyrt.so
   sudo ln -s /usr/local/rpp/lib/fw/pyfwgraphs.cpython-311-x86_64-linux-gnu.so pyfwgraphs.so
   ```

### Required RPP Components

1. **SDK** 
azurengine_sw_v1.6.12.3_x86_Ubuntu.run

## Quick Start

### 1. Start the Server

Start the OpenAI-compatible API server:

```bash
# Daemon mode (background process)
python rpp_openai_server.py -g ~/model_zoo/LLM/Qwen3/qwen3_8b_graph/ \
    --server-port 8001 \
    --server-name 127.0.0.1 \
    -daemon 1

# Foreground mode (for debugging)
python rpp_openai_server.py -g ~/model_zoo/LLM/Qwen3/qwen3_8b_graph/ \
    --server-port 8001 \
    --server-name 127.0.0.1 \
    -daemon 0
```

**Server Options:**
- `-g, --graph_path`: Path to the model graph directory (required)
- `--server-port`: Server port (default: 8001)
- `--server-name`: Server hostname (default: 127.0.0.1, use 0.0.0.0 for network access)
- `-daemon, --daemon_mode`: Run as daemon (1) or foreground (0), default: 1
- `-i, --input_size`: Maximum input size (default: 8192)
- `-t, --target_len`: Maximum target length (default: 8192)
- `-d, --do_sample`: Enable sampling (default: 1)
- `-p, --perf_mode`: Performance mode (default: 0)
- `-l, --low_power`: Low power mode (default: 0)
- `-w, --write_file`: Write output to file (default: 0)
- `-f, --prefix`: Prefix mode (default: 1)

**Daemon Mode:**
- When `-daemon 1`, the server runs as a background process
- Logs are written to `/tmp/daemon-openai.log`
- PID file is stored at `/tmp/daemon-openai.pid`
- To stop: `kill -15 $(cat /tmp/daemon-openai.pid)`
- To view logs: `tail -f /tmp/daemon-openai.log`

### 2. Run the Client

#### Text Generation (Qwen2.5/Qwen3)

```bash
python rpp_openai_client.py
```

The client example demonstrates:
- Text generation with system prompts
- Function calling (tools)
- Streaming responses

#### Vision-Language (Qwen2.5VL)

```bash
# Basic usage with image and prompt
python rpp_openai_vl_client.py \
    --image ./images/man-9581593_640.jpg \
    --prompt "Describe this image"

# Streaming response
python rpp_openai_vl_client.py \
    --image ./images/man-9581593_640.jpg \
    --prompt "What's in this image?" \
    --stream

# Use image URL
python rpp_openai_vl_client.py \
    --image-url "https://example.com/image.jpg" \
    --prompt "Describe this image"

# Use file path directly (if server has file access)
python rpp_openai_vl_client.py \
    --image ./images/man-9581593_640.jpg \
    --prompt "Describe this image" \
    --use-file-path
```
![test_image](./images/man-9581593_640.jpg)

**Response**
    * The image shows a person relaxing outdoors by the side of a body of water, possibly a lake or river. The individual is lying on their back on a blanket spread out on the grass near a tree trunk. They are wearing glasses and an orange-striped shirt paired with dark shorts. The person has a book open in front of them, suggesting they might be reading. There's also a water bottle placed nearby for refreshment. The setting appears peaceful and sunny, ideal for leisurely activities like reading or simply enjoying nature.

**VL Client Options:**
- `--image`: Path to local image file
- `--image-url`: URL of the image (alternative to --image)
- `--prompt`: Text prompt/question about the image (required)
- `--stream`: Enable streaming response
- `--use-file-path`: Use file path directly instead of base64 encoding
- `--api-base`: API base URL (default: http://localhost:8001/v1)
- `--model`: Model name (default: qwen2.5)

## API Usage

### OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1"
)

# Text generation
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    stream=False
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Vision-Language API

```python
from openai import OpenAI
import base64

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1"
)

# Encode image to base64
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_data}"

# Vision-language request
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": "Describe this image"
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

### Function Calling

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1"
)

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
)
```

## API Endpoints

### `GET /v1/models`
List available models.

**Response:**
```json
{
  "data": [
    {
      "id": "qwen2",
      "object": "model"
    }
  ]
}
```

### `POST /v1/chat/completions`
Create a chat completion.

**Request Body:**
```json
{
  "model": "qwen2.5",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Response (Non-streaming):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen2.5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

**Response (Streaming):**
Server-Sent Events (SSE) format:
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"...","choices":[{"delta":{"content":"!"}}]}
data: [DONE]
```

## Vision-Language Model Support

The Qwen2.5VL model supports multi-modal inputs:

### Supported Image Formats
- **Base64 encoded images**: `data:image/jpeg;base64,<base64_data>`
- **File paths**: Absolute paths accessible by the server
- **HTTP/HTTPS URLs**: Remote image URLs

### Message Format
```json
{
  "role": "user",
  "content": [
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,..."
      }
    },
    {
      "type": "text",
      "text": "Describe this image"
    }
  ]
}
```

### Example Use Cases
- Image description and captioning
- Visual question answering
- Image analysis and understanding
- Multi-modal conversations

## Direct Graph Demo (Non-API)

For direct inference without the API server:

```bash
python rpp_graph_demo.py -g ~/model_zoo/LLM/Qwen3/qwen3_8b_graph/
```

**Options:**
- `-g, --graph_path`: Path to graph directory
- `-pf, --prompt_file`: Path to prompt file (default: ./prompts/prompt.txt)
- `-i, --input_size`: Maximum input size (default: 8192)
- `-t, --target_len`: Maximum target length (default: 8192)
- `-d, --do_sample`: Enable sampling (default: 1)
- `-p, --perf_mode`: Performance mode (default: 0)
- `-l, --low_power`: Low power mode (default: 0)
- `-w, --write_file`: Write output to file (default: 0)
- `-f, --prefix`: Prefix mode (default: 1)
- `-r, --run_mode`: Run mode (default: 0)

## Configuration

### Inference Parameters

Default inference parameters (can be customized in code):
- `penalty`: 1.05 (repetition penalty)
- `top_k`: 40
- `top_p`: 0.9
- `temperature`: 0.2
- `min_tokens_to_keep`: 1
- `do_sample`: 1 (from command line)

### Server Configuration

- **Host**: Use `127.0.0.1` for local-only access, `0.0.0.0` for network access
- **Port**: Default 8001, can be changed via `--server-port`
- **CORS**: Enabled for all origins by default
- **Workers**: Single worker (uvicorn workers=1)

## Troubleshooting

### Server Issues

1. **Model not supported error**
   - Check that the graph directory contains a valid model
   - Verify the model type is in the supported list
   - Check `g_version.bin` or `firmware.pb` exists in the graph directory

2. **Port already in use**
   - Change the port: `--server-port 8002`
   - Kill existing process: `lsof -ti:8001 | xargs kill`

3. **Daemon not starting**
   - Check logs: `tail -f /tmp/daemon-openai.log`
   - Verify RPP libraries are accessible
   - Check shared memory: `/dev/shm/rpp_dev_shared_mem`

### Client Issues

1. **Connection refused**
   - Verify server is running: `curl http://localhost:8001/v1/models`
   - Check server host and port match client configuration

2. **Empty response**
   - Check server logs for errors
   - Verify model loaded successfully
   - For VL models, ensure image format is correct

3. **Vision model errors**
   - Ensure image file exists and is readable
   - Check base64 encoding is correct
   - Verify image format is supported (JPEG, PNG, etc.)

## Examples

### Example 1: Text Generation with System Prompt

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)
print(response.choices[0].message.content)
```

### Example 2: Streaming Response

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

stream = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Example 3: Vision-Language with Base64 Image

```python
from openai import OpenAI
import base64

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

# Load and encode image
with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_data}"

# Create vision-language request
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "What objects are in this image?"}
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

## License

See the main project license.

## Support

For issues and questions, please refer to the project documentation or contact the development team.
