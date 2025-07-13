# -*- coding: utf-8 -*-
"""
Tokenizer conversion command
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers_converter.tokenizers.ernie4_5_converter import Ernie45Converter
from tokenizers_converter.tokenizers.baichuan_converter import BaichuanConverter


def get_available_converters():
    """Get available converter list"""
    return {
        "ernie4.5": Ernie45Converter,
        "baichuan": BaichuanConverter,
    }


def convert_tokenizer(
    pretrained_model_name_or_path: str,
    output_path: str,
    converter_type: str,
    vocab_size: Optional[int] = None
):
    """
    Convert tokenizer
    
    Args:
        pretrained_model_name_or_path: HuggingFace model name or local path
        output_path: Output path
        converter_type: Converter type
        vocab_size: Vocabulary size
    """
    converters = get_available_converters()
    
    if converter_type not in converters:
        available = ", ".join(converters.keys())
        raise ValueError(f"Unsupported converter type '{converter_type}', available types: {available}")
    
    print(f"Loading model: {pretrained_model_name_or_path}")
    
    # Load original tokenizer (supports both HuggingFace model names and local paths)
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    except Exception as e:
        print(f"Error: Cannot load tokenizer '{pretrained_model_name_or_path}': {e}")
        print("Note: This can be either a HuggingFace model name (e.g., 'bert-base-uncased') or a local path")
        sys.exit(1)
    
    print(f"Using converter: {converter_type}")

    # Get converter class
    converter_class = converters[converter_type]
    
    # Execute conversion
    try:
        # Create converter with original tokenizer
        converter = converter_class(tokenizer)
        converted_tokenizer_object = converter.converted()
        
        # Create PreTrainedTokenizerFast with all necessary configurations
        converted_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=converted_tokenizer_object,
            model_input_names=tokenizer.model_input_names,
            model_max_length=tokenizer.model_max_length,
            clean_up_tokenization_spaces=False,
            # Pass special tokens from original tokenizer
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            unk_token=tokenizer.unk_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.pad_token,
            cls_token=tokenizer.cls_token,
            mask_token=tokenizer.mask_token,
            additional_special_tokens=tokenizer.additional_special_tokens,
        )

        converted_tokenizer.chat_template = tokenizer.chat_template
        
        # Set vocabulary size
        if vocab_size is not None:
            print(f"Setting vocabulary size: {vocab_size}")
            # Logic for adjusting vocabulary size can be added here
        
        # Save converted tokenizer using save_pretrained
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_tokenizer.save_pretrained(str(output_dir))
        print(f"Conversion completed, saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: Conversion failed: {e}")
        sys.exit(1)


def main():
    """Main function for convert subcommand"""
    parser = argparse.ArgumentParser(
        description="Convert tokenizer format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Baichuan tokenizer using HuggingFace model name
  tokenizers-converter convert --pretrained_model_name_or_path baichuan-inc/Baichuan2-7B-Chat --output-path ./output --type baichuan
  
  # Convert ERNIE tokenizer using HuggingFace model name
  tokenizers-converter convert --pretrained_model_name_or_path baidu/ernie-4.5-base --output-path ./output --type ernie4.5
  
  # Convert using local path
  tokenizers-converter convert --pretrained_model_name_or_path ./baichuan-model --output-path ./output --type baichuan
  
  # Use short arguments
  tokenizers-converter convert -m baichuan-inc/Baichuan2-13B-Chat -o ./output -t baichuan --vocab-size 50000
        """
    )
    
    parser.add_argument(
        "--pretrained_model_name_or_path", "-m",
        required=True,
        help="HuggingFace model name or local model path"
    )
    
    parser.add_argument(
        "--output-path", "-o", 
        required=True,
        help="Output path"
    )
    
    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=list(get_available_converters().keys()),
        help="Converter type"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size (optional)"
    )
    
    args = parser.parse_args()
    
    # For local paths, validate that they exist
    if os.path.exists(args.pretrained_model_name_or_path):
        print(f"Using local model path: {args.pretrained_model_name_or_path}")
    else:
        print(f"Using HuggingFace model name: {args.pretrained_model_name_or_path}")
    
    # Execute conversion
    convert_tokenizer(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_path=args.output_path,
        converter_type=args.type,
        vocab_size=args.vocab_size
    )


if __name__ == "__main__":
    main() 