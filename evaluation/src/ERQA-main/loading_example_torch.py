#!/usr/bin/env python3

import tfrecord
import io
from PIL import Image
import numpy as np

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

def main():
    # Path to the TFRecord file
    tfrecord_path = './data/erqa.tfrecord'
    
    # Number of examples to display
    num_examples = 3
    
    print(f"Loading first {num_examples} examples from {tfrecord_path}...")
    print("-" * 50)
    
    # Process examples using tfrecord.tfrecord_iterator
    for i, example_proto in enumerate(tfrecord.tfrecord_iterator(tfrecord_path)):
        if i >= num_examples:
            break
            
        # Parse the example without TensorFlow
        example = parse_example_without_tf(example_proto)
        
        # Extract data from example
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
        
        # Display image dimensions for each image
        for j, img_encoded in enumerate(images_encoded):
            # Decode the image using Pillow from bytes stream
            img_stream = io.BytesIO(img_encoded)
            img = Image.open(img_stream)
            img_array = np.array(img)
            
            print(f"  Image {j+1} dimensions: {img_array.shape}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()