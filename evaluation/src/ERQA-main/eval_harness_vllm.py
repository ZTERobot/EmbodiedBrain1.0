import os
import numpy as np
from PIL import Image
import io
import time
import argparse
import sys
import base64
from collections import defaultdict
import tfrecord
import requests

# Parse TFRecord example
def parse_example_without_tf(example_proto):
    """
    Parse a TFRecord example protocol buffer without using TensorFlow.
    """
    # Create an Example object from the protocol buffer
    example = tfrecord.example_pb2.Example()
    example.ParseFromString(example_proto)

    features = {}

    # Get the features from the example
    for key, value in example.features.feature.items():
        if value.bytes_list.value:
            # Handle string/bytes data
            features[key] = [v for v in value.bytes_list.value]
        elif value.int64_list.value:
            # Handle integer data
            features[key] = [v for v in value.int64_list.value]
        elif value.float_list.value:
            # Handle float data
            features[key] = [v for v in value.float_list.value]
    
    # Manually handle the single-value and multiple-value features
    parsed_features = {
        'answer': features.get('answer', [b''])[0].decode('utf-8'),
        'image/encoded': features.get('image/encoded', []),
        'question_type': features.get('question_type', [b'']),
        'visual_indices': features.get('visual_indices', []),
        'question': features.get('question', [b''])[0].decode('utf-8'),
    }

    return parsed_features


# Convert TF tensor image to PIL Image
def tensor_to_pil(image_tensor):
    """Convert a TensorFlow image tensor to a PIL Image."""
    if isinstance(image_tensor, bytes):
        return Image.open(io.BytesIO(image_tensor))
    else:
        # If it's a numpy array
        return Image.fromarray(image_tensor.astype('uint8'))

def query_vllm(url, model_name, contents, max_tokens=300, connection_retries=5):
    """
    Query the vLLM API with a question and images, with retry logic.

    This function replaces the OpenAI API client with a direct HTTP call
    to a vLLM server endpoint.

    Args:
        api_endpoint: The URL of the vLLM API endpoint (e.g., "http://localhost:8000/v1/chat/completions").
        model_name: Name of the model to use (e.g., "llama-3-8b").
        contents: List containing the question segments and images in the correct order.
        max_tokens: Maximum number of tokens in the response.
        connection_retries: Maximum number of retries for connection errors.

    Returns:
        The JSON response from the vLLM API, or None if the request fails after all retries.
    """
    # Convert contents to the chat completion message format
    # Note: For multimodal models like LLaVA, vLLM's API might require a different
    # format for images (e.g., a local file path), and may not support base64 out-of-the-box.
    # This part of the code assumes the vLLM server is capable of processing
    # base64 image URLs.
    message_content = []
    
    for item in contents:
        if isinstance(item, str):
            message_content.append({
                "type": "text",
                "text": item
            })
        else:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            item.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })
    
    # Construct the JSON payload for the vLLM API
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    
    connection_retry_count = 0
    
    while connection_retry_count < connection_retries:
        try:
            # Send the request to the vLLM API endpoint
            response = requests.post(
                url,
                json=payload,
                timeout=30 # Set a timeout to prevent indefinite waiting
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            
            return response.json()
            
        except requests.exceptions.ConnectionError as e:
            connection_retry_count += 1
            print(f"Connection error detected. Retry {connection_retry_count}/{connection_retries}. Error: {e}")
            
            if connection_retry_count >= connection_retries:
                print(f"Maximum connection retries ({connection_retries}) reached. Exiting.")
                return None
            
            # Use fixed 2-second backoff
            print("Waiting 2 seconds before retrying...")
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            # Catch other request-related errors (e.g., HTTP status codes, timeouts)
            print(f"Error querying vLLM API: {e}")
            return None

# Custom exception for resource exhaustion
class ResourceExhaustedError(Exception):
    pass

# Print evaluation summary
def print_summary(total_examples, correct_examples, single_image_total, single_image_correct, 
                 multi_image_total, multi_image_correct, question_type_stats):
    """Print the evaluation summary statistics."""
    print("\n=== Evaluation Summary ===")
    print(f"Total examples: {total_examples}")
    
    if total_examples > 0:
        print(f"Overall accuracy: {correct_examples/total_examples:.2%} ({correct_examples}/{total_examples})")
    else:
        print("No examples processed")
    
    if single_image_total > 0:
        print(f"Single-image accuracy: {single_image_correct/single_image_total:.2%} ({single_image_correct}/{single_image_total})")
    else:
        print("No single-image examples processed")
    
    if multi_image_total > 0:
        print(f"Multi-image accuracy: {multi_image_correct/multi_image_total:.2%} ({multi_image_correct}/{multi_image_total})")
    else:
        print("No multi-image examples processed")
    
    # Print accuracy by question type
    if question_type_stats:
        print("\n--- Accuracy by Question Type ---")
        for q_type, stats in sorted(question_type_stats.items()):
            total = stats['total']
            correct = stats['correct']
            if total > 0:
                print(f"{q_type}: {correct/total:.2%} ({correct}/{total})")
            else:
                print(f"{q_type}: No examples")

def main():
    parser = argparse.ArgumentParser(description='Multimodal API Evaluation Harness')
    parser.add_argument('--url', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--tfrecord_path', type=str, default='./data/erqa.tfrecord',
                        help='Path to the TFRecord file')
    parser.add_argument('--num_examples', type=int, default=1,
                        help='Number of examples to process')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Maximum number of retries per API key on resource exhaustion (default: 2)')
    parser.add_argument('--max_tokens', type=int, default=300,
                        help='Maximum number of tokens in the response (for OpenAI only)')
    parser.add_argument('--connection_retries', type=int, default=5,
                        help='Maximum number of retries for connection errors (for OpenAI only, default: 5)')
    
    args = parser.parse_args()
    
    
    # Load TFRecord dataset
    # dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    # dataset = dataset.map(parse_example_without_tf)
    
    # Initialize counters for tracking accuracy
    total_examples = 0
    correct_examples = 0
    single_image_total = 0
    single_image_correct = 0
    multi_image_total = 0
    multi_image_correct = 0
    
    # Track accuracy by question type
    question_type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # Track the last successful client index
    last_successful_client_idx = 0

    # 使用 tfrecord.tfrecord_iterator 替代 tf.data.TFRecordDataset
    # Process examples
    try:
        for i, example_proto in enumerate(tfrecord.tfrecord_iterator(args.tfrecord_path)):
        # for i, example in enumerate(dataset.take(args.num_examples)):
            # Extract data from example

            # answer = example['answer'].numpy().decode('utf-8')
            # images_encoded = example['image/encoded'].numpy()
            # question_type = example['question_type'][0].numpy().decode('utf-8') if len(example['question_type']) > 0 else "Unknown"
            # visual_indices = example['visual_indices'].numpy()
            # question = example['question'].numpy().decode('utf-8')

            example = parse_example_without_tf(example_proto)
            answer = example['answer']
            images_encoded = example['image/encoded']
            question_type = example['question_type'][0].decode('utf-8') if example['question_type'] else "Unknown"
            visual_indices = np.array(example['visual_indices'])
            question = example['question']
            
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {question}")
            print(f"Question Type: {question_type}")
            print(f"Ground Truth Answer: {answer}")
            print(f"Number of images: {len(images_encoded)}")
            print(f"Visual indices: {visual_indices}")
            print(f"Starting with API key {last_successful_client_idx+1}")
            
            # Convert encoded images to PIL images
            pil_images = []
            for img_encoded in images_encoded:
                # Decode the image tensor
                # img_tensor = tf.io.decode_image(img_encoded).numpy()
                # pil_img = Image.fromarray(img_tensor)

                img_stream = io.BytesIO(img_encoded)
                pil_img = Image.open(img_stream)
                pil_images.append(pil_img)
            
            # Prepare contents for API based on visual_indices
            # Create a list of (image, index) pairs
            image_index_pairs = list(zip(pil_images, visual_indices))
            
            # Sort by visual_indices
            image_index_pairs.sort(key=lambda x: x[1])
            
            # Split the question text and interleave with images
            contents = []
            
            # Handle case where visual_indices is empty (place images at the beginning)
            if len(visual_indices) == 0:
                # Add all images at the beginning
                for img in pil_images:
                    contents.append(img)
                # Then add the question text
                contents.append(question)
            # Handle case where all indices are 0 (all images at the beginning)
            elif all(idx == 0 for idx in visual_indices):
                # First add all images
                for img, _ in image_index_pairs:
                    contents.append(img)
                # Then add the question text
                contents.append(question)
            else:
                # Split question at visual_indices positions
                last_pos = 0
                
                # Process each image and its position
                for img, idx in image_index_pairs:
                    if idx == 0:
                        # Image goes at the beginning
                        contents.append(img)
                    else:
                        # Add text segment before this image
                        if idx <= len(question):
                            text_segment = question[last_pos:idx]
                            if text_segment:
                                contents.append(text_segment)
                            contents.append(img)
                            last_pos = idx
                        else:
                            # If index is beyond question length, just append the image
                            contents.append(img)
                
                # Add any remaining text
                if last_pos < len(question):
                    contents.append(question[last_pos:])
                
                # If no content was added (e.g., all indices were beyond question length),
                # add the full question at the beginning
                if not contents:
                    contents.append(question)
                    for img, _ in image_index_pairs:
                        contents.append(img)
            
            # Print the content structure for debugging
            content_structure = []
            for item in contents:
                if isinstance(item, str):
                    content_structure.append(f"Text: '{item}'")
                else:
                    content_structure.append("Image")
            print(f"Content structure: {content_structure}")
            
            # Query API with retry logic, starting with the last successful client
            print(f"Querying VLLM API...")
            start_time = time.time()
            
            response_json = query_vllm(args.url, args.model, contents, args.max_tokens, args.max_retries)
            
            end_time = time.time()
            
            # Process response
            if response_json:
                # response_text = response.choices[0].message.content
                response_text = response_json['choices'][0]['message']['content']
                response_text = response_text.replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
                print(f"VLLM API Response: {response_text}")
                print(f"Response time: {end_time - start_time:.2f} seconds")
                
                # Check if the answer is correct (exact match)
                is_correct = response_text.replace(".", "").strip().lower() == answer.strip().lower()
                
                # Update counters
                total_examples += 1
                if is_correct:
                    correct_examples += 1
                    print("✓ Correct answer (exact match)")
                else:
                    print("✗ Incorrect answer (based on exact match)")
                
                # Track single vs multi-image accuracy
                if len(images_encoded) == 1:
                    single_image_total += 1
                    if is_correct:
                        single_image_correct += 1
                else:
                    multi_image_total += 1
                    if is_correct:
                        multi_image_correct += 1
                
                # Track accuracy by question type
                question_type_stats[question_type]['total'] += 1
                if is_correct:
                    question_type_stats[question_type]['correct'] += 1
            else:
                print(f"Failed to get response from VLLM API")
            
            print("-" * 50)
    
    except ResourceExhaustedError:
        # We've hit a resource exhaustion error with all API keys, exit early but still print summary
        print("\nExiting early due to all API keys being exhausted.")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    finally:
        # Always print summary, even if we exit early
        print_summary(total_examples, correct_examples, single_image_total, single_image_correct, 
                     multi_image_total, multi_image_correct, question_type_stats)

if __name__ == "__main__":
    main() 