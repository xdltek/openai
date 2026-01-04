#!/usr/bin/env python3
"""
Client example for Qwen VL (Vision-Language) model using OpenAI-compatible API.

Usage:
    python rpp_openai_vl_client.py --image path/to/image.jpg --prompt "Describe this image"
    python rpp_openai_vl_client.py --image path/to/image.jpg --prompt "What's in this image?" --stream
"""

from openai import OpenAI
import base64
import argparse
import os
import sys

# Configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 data URL."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image format from file extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.jpg' or ext == '.jpeg':
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        elif ext == '.gif':
            mime_type = 'image/gif'
        elif ext == '.webp':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # default
        
        return f"data:{mime_type};base64,{base64_image}"

def create_vl_message(image_path: str, prompt: str, use_base64: bool = True):
    """
    Create a message for vision-language model.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt/question about the image
        use_base64: If True, encode image as base64. If False, use file path directly.
    
    Returns:
        List of messages in OpenAI API format
    """
    if use_base64:
        # Encode image as base64 (recommended for remote servers)
        image_url = encode_image_to_base64(image_path)
    else:
        # Use file path directly (works if server has access to the file)
        image_url = os.path.abspath(image_path)
    
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
                    "text": prompt
                }
            ]
        }
    ]
    
    return messages

def main():
    parser = argparse.ArgumentParser(
        description='Qwen VL (Vision-Language) model client using OpenAI-compatible API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with image and prompt
  python rpp_openai_vl_client.py --image ./image.jpg --prompt "Describe this image"
  
  # Use file path instead of base64 (if server has file access)
  python rpp_openai_vl_client.py --image ./image.jpg --prompt "What's in this image?" --use-file-path
  
  # Streaming response
  python rpp_openai_vl_client.py --image ./image.jpg --prompt "Describe this image" --stream
  
  # Use image URL
  python rpp_openai_vl_client.py --image-url "https://example.com/image.jpg" --prompt "Describe this image"
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--image-url', type=str, help='URL of the image (alternative to --image)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt/question about the image')
    parser.add_argument('--stream', action='store_true', help='Enable streaming response')
    parser.add_argument('--use-file-path', action='store_true', 
                       help='Use file path directly instead of base64 encoding (requires server file access)')
    parser.add_argument('--api-base', type=str, default=openai_api_base,
                       help=f'API base URL (default: {openai_api_base})')
    parser.add_argument('--model', type=str, default='qwen2.5',
                       help='Model name (default: qwen2.5)')
    
    args = parser.parse_args()
    
    # Validate image input
    if not args.image and not args.image_url:
        parser.error("Either --image or --image-url must be provided")
    
    if args.image and args.image_url:
        parser.error("Cannot specify both --image and --image-url")
    
    # Check if image file exists
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Create OpenAI client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=args.api_base,
    )
    
    # Create messages
    if args.image:
        messages = create_vl_message(args.image, args.prompt, use_base64=not args.use_file_path)
    else:
        # Use image URL directly
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": args.image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": args.prompt
                    }
                ]
            }
        ]
    
    print(f"Image: {args.image or args.image_url}")
    print(f"Prompt: {args.prompt}")
    print(f"Streaming: {args.stream}")
    print("-" * 50)
    
    try:
        if args.stream:
            # Streaming response
            print("Response: ", end="", flush=True)
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                # Check if chunk has choices
                if not hasattr(chunk, 'choices') or len(chunk.choices) == 0:
                    continue
                
                # Get delta content from streaming response
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n" + "-" * 50)
            print(f"Full response length: {len(full_response)} characters")
            
        else:
            # Non-streaming response
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=False
            )
            
            # For non-streaming, use message.content, not delta.content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    print("Response:")
                    print(message.content)
                else:
                    print("Response: (empty)")
                    print(f"Message object: {message}")
            else:
                print("Error: No choices in response")
                print(f"Response: {response}")
            
            print("\n" + "-" * 50)
            if hasattr(response, 'usage') and response.usage:
                print(f"Usage: {response.usage}")
            
    except Exception as e:
        import traceback
        print(f"\nError: {e}", file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
