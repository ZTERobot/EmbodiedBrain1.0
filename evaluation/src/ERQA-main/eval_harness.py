import os
import numpy as np
from PIL import Image
import io
import time
import argparse
import sys
import base64
from google import genai
from google.genai import types
from collections import defaultdict
from openai import OpenAI

# Configure API key
def configure_genai_api(api_keys=None):
    """
    Configure the Gemini API with the provided keys or from environment variable.
    
    Args:
        api_keys: A single API key string or a list of API key strings
        
    Returns:
        A list of Gemini API clients
    """
    clients = []
    
    # If no keys provided, try to get from environment
    if api_keys is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables and no key provided")
        api_keys = [api_key]
    
    # Convert single key to list
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    
    # Create a client for each API key
    for key in api_keys:
        clients.append(genai.Client(api_key=key))
    
    return clients, api_keys

# Configure OpenAI API
def configure_openai_api(api_keys=None):
    """
    Configure the OpenAI API with the provided keys or from environment variable.
    
    Args:
        api_keys: A single API key string or a list of API key strings
        
    Returns:
        A list of OpenAI API clients
    """
    clients = []
    
    # If no keys provided, try to get from environment
    if api_keys is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables and no key provided")
        api_keys = [api_key]
    
    # Convert single key to list
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    
    # Create a client for each API key
    for key in api_keys:
        clients.append(OpenAI(api_key=key))
    
    return clients, api_keys

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

# Query Gemini API with an example
def query_gemini(clients, api_keys, model_name, contents, max_retries=1, start_client_idx=0):
    """
    Query the Gemini API with a question and images, with retry logic.
    
    Args:
        clients: List of Gemini API clients
        api_keys: List of API keys (for logging purposes)
        model_name: Name of the Gemini model to use
        contents: List containing the question segments and images in the correct order
        max_retries: Maximum number of retries per API key on resource exhaustion
        start_client_idx: Index of the client to start with (for using the last successful key)
        
    Returns:
        Tuple of (response, successful_client_idx) where successful_client_idx is the index
        of the client that successfully processed the request
    """
    # Reorder clients and api_keys to start with the specified index
    ordered_clients = clients[start_client_idx:] + clients[:start_client_idx]
    ordered_api_keys = api_keys[start_client_idx:] + api_keys[:start_client_idx]
    
    for idx, (client, key) in enumerate(zip(ordered_clients, ordered_api_keys)):
        # Calculate the original index for this client
        original_idx = (start_client_idx + idx) % len(clients)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Generate content
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=500,
                        temperature=0.0
                    )
                )
                print(response.text)
                
                # Return the response and the original index of the successful client
                return response, original_idx
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a resource exhaustion error (429)
                if "429 RESOURCE_EXHAUSTED" in error_str:
                    retry_count += 1
                    print(f"Resource exhaustion detected with API key {original_idx+1}. Retry {retry_count}/{max_retries}")
                    
                    if retry_count >= max_retries:
                        print(f"Maximum retries ({max_retries}) reached for API key {original_idx+1}.")
                        # Try the next API key if available
                        break
                    
                    # Use fixed 2-second backoff instead of exponential
                    print("Waiting 2 seconds before retrying...")
                    time.sleep(2)
                else:
                    # For other errors, log and return None
                    print(f"Error querying Gemini API: {error_str}")
                    return None, start_client_idx
    
    # If we've exhausted all API keys and retries
    print("All API keys have reached their quota limits. Exiting.")
    raise ResourceExhaustedError("All API keys exhausted")

# Query OpenAI API with an example
def query_openai(clients, api_keys, model_name, contents, max_tokens=300, max_retries=1, start_client_idx=0, connection_retries=5):
    """
    Query the OpenAI API with a question and images, with retry logic.
    
    Args:
        clients: List of OpenAI API clients
        api_keys: List of API keys (for logging purposes)
        model_name: Name of the OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
        contents: List containing the question segments and images in the correct order
        max_tokens: Maximum number of tokens in the response
        max_retries: Maximum number of retries per API key on resource exhaustion
        start_client_idx: Index of the client to start with (for using the last successful key)
        connection_retries: Maximum number of retries for connection errors
        
    Returns:
        Tuple of (response, successful_client_idx) where successful_client_idx is the index
        of the client that successfully processed the request
    """
    # Reorder clients and api_keys to start with the specified index
    ordered_clients = clients[start_client_idx:] + clients[:start_client_idx]
    ordered_api_keys = api_keys[start_client_idx:] + api_keys[:start_client_idx]
    
    # Convert contents to OpenAI format
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
    
    for idx, (client, key) in enumerate(zip(ordered_clients, ordered_api_keys)):
        # Calculate the original index for this client
        original_idx = (start_client_idx + idx) % len(clients)
        retry_count = 0
        
        while retry_count < max_retries:
            # Initialize connection retry counter
            connection_retry_count = 0
            
            while connection_retry_count < connection_retries:
                try:
                    # Generate content
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": message_content
                            }
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens
                    )
                    
                    # Return the response and the original index of the successful client
                    return response, original_idx
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if this is a connection error
                    if "Connection error" in error_str:
                        connection_retry_count += 1
                        print(f"Connection error detected with API key {original_idx+1}. Retry {connection_retry_count}/{connection_retries}")
                        
                        if connection_retry_count >= connection_retries:
                            print(f"Maximum connection retries ({connection_retries}) reached for API key {original_idx+1}.")
                            # Instead of exiting fatally, break out of the connection retry loop
                            # to try the next API key if available
                            break
                        
                        # Use fixed 2-second backoff
                        print("Waiting 2 seconds before retrying...")
                        time.sleep(2)
                    # Check if this is a rate limit error (429)
                    elif "429" in error_str:
                        retry_count += 1
                        print(f"Rate limit detected with API key {original_idx+1}. Retry {retry_count}/{max_retries}")
                        
                        if retry_count >= max_retries:
                            print(f"Maximum retries ({max_retries}) reached for API key {original_idx+1}.")
                            # Try the next API key if available
                            break
                        
                        # Use fixed 2-second backoff instead of exponential
                        print("Waiting 2 seconds before retrying...")
                        time.sleep(2)
                        # Break out of the connection retry loop to go to the rate limit retry loop
                        break
                    else:
                        # For other errors, log and return None
                        print(f"Error querying OpenAI API: {error_str}")
                        return None, start_client_idx
            
            # If we've exhausted connection retries, break out of the rate limit retry loop
            # to try the next API key
            if connection_retry_count >= connection_retries:
                break
    
    # If we've exhausted all API keys and retries
    print("All API keys have reached their quota limits or encountered persistent connection errors. Exiting.")
    raise ResourceExhaustedError("All API keys exhausted")

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
    parser.add_argument('--tfrecord_path', type=str, default='./data/erqa.tfrecord',
                        help='Path to the TFRecord file')
    parser.add_argument('--api', type=str, choices=['gemini', 'openai'], default='gemini',
                        help='API to use: gemini or openai')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name to use (defaults: gemini-2.0-flash-exp for Gemini, gpt-4o for OpenAI). '
                             'Available Gemini models include: gemini-2.0-flash-exp, gemini-2.0-pro, gemini-2.0-pro-exp-02-05')
    parser.add_argument('--gemini_api_key', type=str, default=None, action='append',
                        help='Gemini API key (can be specified multiple times for multiple keys)')
    parser.add_argument('--openai_api_key', type=str, default=None, action='append',
                        help='OpenAI API key (can be specified multiple times for multiple keys)')
    parser.add_argument('--api_keys_file', type=str, default=None,
                        help='Path to a file containing API keys (one per line, format: "gemini:KEY" or "openai:KEY")')
    parser.add_argument('--num_examples', type=int, default=1,
                        help='Number of examples to process')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Maximum number of retries per API key on resource exhaustion (default: 2)')
    parser.add_argument('--max_tokens', type=int, default=300,
                        help='Maximum number of tokens in the response (for OpenAI only)')
    parser.add_argument('--connection_retries', type=int, default=5,
                        help='Maximum number of retries for connection errors (for OpenAI only, default: 5)')
    
    args = parser.parse_args()
    
    # Set default model based on API
    if args.model is None:
        if args.api == 'gemini':
            args.model = 'gemini-2.0-flash-exp'
        else:  # openai
            args.model = 'gpt-4o'
    
    # Collect API keys from all sources
    gemini_api_keys = []
    openai_api_keys = []
    
    # Add keys from command line arguments
    if args.gemini_api_key:
        gemini_api_keys.extend(args.gemini_api_key)
    
    if args.openai_api_key:
        openai_api_keys.extend(args.openai_api_key)
    
    # Add keys from file if specified
    if args.api_keys_file and os.path.exists(args.api_keys_file):
        with open(args.api_keys_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if ':' in line:
                    api_type, key = line.split(':', 1)
                    if api_type.lower() == 'gemini':
                        gemini_api_keys.append(key)
                    elif api_type.lower() == 'openai':
                        openai_api_keys.append(key)
                else:
                    # If no prefix, assume it's for the selected API
                    if args.api == 'gemini':
                        gemini_api_keys.append(line)
                    else:
                        openai_api_keys.append(line)
    
    # If no keys provided, try environment variable
    if args.api == 'gemini' and not gemini_api_keys:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            gemini_api_keys = [env_key]
    
    if args.api == 'openai' and not openai_api_keys:
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            openai_api_keys = [env_key]
    
    # Configure API clients
    if args.api == 'gemini':
        clients, api_keys = configure_genai_api(gemini_api_keys)
        print(f"Configured {len(clients)} Gemini API key(s)")
    else:  # openai
        clients, api_keys = configure_openai_api(openai_api_keys)
        print(f"Configured {len(clients)} OpenAI API key(s)")
    
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
            print(f"Querying {args.api.capitalize()} API...")
            start_time = time.time()
            
            if args.api == 'gemini':
                response_tuple = query_gemini(clients, api_keys, args.model, contents, args.max_retries, last_successful_client_idx)
            else:  # openai
                response_tuple = query_openai(clients, api_keys, args.model, contents, args.max_tokens, args.max_retries, last_successful_client_idx, args.connection_retries)
            
            if response_tuple:
                response, successful_client_idx = response_tuple
                # Update the last successful client index for the next query
                last_successful_client_idx = successful_client_idx
                print(f"Successfully used API key {successful_client_idx+1}")
            else:
                response = None
            
            end_time = time.time()
            
            # Process response
            if response:
                if args.api == 'gemini':
                    response_text = response.text
                else:  # openai
                    response_text = response.choices[0].message.content
                
                print(f"{args.api.capitalize()} Response: {response_text}")
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
                print(f"Failed to get response from {args.api.capitalize()} API")
            
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