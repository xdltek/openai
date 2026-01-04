#!/usr/bin/env python3
"""
Simple client example for Qwen VL (Vision-Language) model.
This is a simpler version that shows basic usage.
"""

from openai import OpenAI
import base64
import os

# Configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

# Create OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Example 1: Using base64 encoded image
def example_with_base64_image():
    """Example using base64 encoded image."""
    # Path to your image
    image_path = "./images/man-9581593_640.jpg"  # Change this to your image path
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")
        return
    
    # Encode image to base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
    
    # Create messages with image and prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ]
        }
    ]
    
    print("Sending request with base64 image...")
    print(f"Image: {image_path}")
    print(f"Prompt: Describe this image in detail.")
    print("-" * 50)
    
    # Send request (non-streaming)
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        stream=False
    )
    
    print("Response:")
    print(response.choices[0].message.content)
    print(f"\nUsage: {response.usage}")

# Example 2: Using image file path (if server has file access)
def example_with_file_path():
    """Example using file path directly."""
    # Path to your image (absolute path recommended)
    image_path = os.path.abspath("./image/man-9581593_640.jpg")  # Change this to your image path
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print("Please update the image_path variable with a valid image path.")
        return
    
    # Create messages with image path and prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path  # Use file path directly
                    }
                },
                {
                    "type": "text",
                    "text": "What can you see in this image?"
                }
            ]
        }
    ]
    
    print("Sending request with file path...")
    print(f"Image: {image_path}")
    print(f"Prompt: What can you see in this image?")
    print("-" * 50)
    
    # Send request (non-streaming)
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        stream=False
    )
    
    print("Response:")
    print(response.choices[0].message.content)
    print(f"\nUsage: {response.usage}")

# Example 3: Using image URL
def example_with_image_url():
    """Example using image URL."""
    # Image URL
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    # Create messages with image URL and prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Tell me about this image."
                }
            ]
        }
    ]
    
    print("Sending request with image URL...")
    print(f"Image URL: {image_url}")
    print(f"Prompt: Tell me about this image.")
    print("-" * 50)
    
    # Send request (non-streaming)
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        stream=False
    )
    
    print("Response:")
    print(response.choices[0].message.content)
    print(f"\nUsage: {response.usage}")

# Example 4: Streaming response
def example_streaming():
    """Example with streaming response."""
    # Path to your image
    image_path = "./image/man-9581593_640.jpg"  # Change this to your image path
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Encode image to base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
    
    # Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ]
        }
    ]
    
    print("Sending streaming request...")
    print(f"Image: {image_path}")
    print(f"Prompt: Describe this image.")
    print("-" * 50)
    print("Response: ", end="", flush=True)
    
    # Send streaming request
    for chunk in client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

if __name__ == '__main__':
    print("=" * 50)
    print("Qwen VL Client Examples")
    print("=" * 50)
    print()
    
    # Run example 1: Base64 image
    print("Example 1: Using base64 encoded image")
    print("=" * 50)
    example_with_base64_image()
    print()
    
    # Uncomment to run other examples:
    # print("Example 2: Using file path")
    # print("=" * 50)
    # example_with_file_path()
    # print()
    
    # print("Example 3: Using image URL")
    # print("=" * 50)
    # example_with_image_url()
    # print()
    
    # print("Example 4: Streaming response")
    # print("=" * 50)
    # example_streaming()
    # print()

