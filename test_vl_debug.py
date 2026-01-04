#!/usr/bin/env python3
"""
Debug script to test Qwen VL model integration
"""

import json
import traceback
from openai import OpenAI

# Configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

# Create OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Test 1: Simple text message (should work)
print("=" * 60)
print("Test 1: Simple text message")
print("=" * 60)
try:
    messages = [
        {
            "role": "user",
            "content": "Hello, can you see this message?"
        }
    ]
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        stream=False
    )
    print("SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n")

# Test 2: Message with image URL (file path)
print("=" * 60)
print("Test 2: Message with image file path")
print("=" * 60)
try:
    # Use absolute path
    import os
    image_path = os.path.abspath("./images/man-9581593_640.jpg")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ]
        }
    ]
    
    print(f"Image path: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        stream=False
    )
    print("SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n")

# Test 3: Message with base64 image
print("=" * 60)
print("Test 3: Message with base64 image")
print("=" * 60)
try:
    import base64
    import os
    
    image_path = "./images/man-9581593_640.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
    else:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_data}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url  # Full base64 URL
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }
        ]
        
        print(f"Base64 URL length: {len(image_url)}")
        print("Sending request...")
        
        response = client.chat.completions.create(
            model="qwen2.5",
            messages=messages,
            stream=False
        )
        print("SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n")
print("=" * 60)
print("Debug tests completed")
print("=" * 60)

