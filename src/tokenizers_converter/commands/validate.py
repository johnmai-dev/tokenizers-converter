# -*- coding: utf-8 -*-
"""
Validate tokenizer command
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str):
    """
    Load tokenizer
    
    Args:
        tokenizer_path: Tokenizer path
        
    Returns:
        Loaded tokenizer object
    """
    path = Path(tokenizer_path)
    
    if path.is_file():
        # If it's a file, try to load as tokenizer.json
        if path.suffix == '.json':
            return Tokenizer.from_file(str(path))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    elif path.is_dir():
        # If it's a directory, try to load as HuggingFace model
        try:
            return AutoTokenizer.from_pretrained(str(path))
        except Exception as e:
            # If failed, try to load tokenizer.json in the directory
            json_path = path / "tokenizer.json"
            if json_path.exists():
                return Tokenizer.from_file(str(json_path))
            else:
                raise ValueError(f"Cannot find valid tokenizer file in directory: {e}")
    else:
        raise ValueError(f"Path does not exist: {tokenizer_path}")


def validate_tokenizer_basic(tokenizer, tokenizer_path: str):
    """
    Basic validation
    
    Args:
        tokenizer: Tokenizer object
        tokenizer_path: Tokenizer path
    """
    print(f"Validating tokenizer: {tokenizer_path}")
    print("=" * 50)
    
    # Check tokenizer type
    tokenizer_type = type(tokenizer).__name__
    print(f"Tokenizer type: {tokenizer_type}")
    
    # Check vocabulary size
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab_size'):
        vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = "Unknown"
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Check special tokens
    if hasattr(tokenizer, 'special_tokens_map'):
        special_tokens = tokenizer.special_tokens_map
        print(f"Special tokens: {special_tokens}")
    
    return True


def validate_tokenizer_functionality(tokenizer, test_texts: List[str]):
    """
    Functionality validation
    
    Args:
        tokenizer: Tokenizer object
        test_texts: List of test texts
    """
    print("\nFunctionality validation:")
    print("-" * 30)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest text {i}: {text}")
        
        try:
            # Encoding test
            if hasattr(tokenizer, 'encode'):
                if isinstance(tokenizer, Tokenizer):
                    # HuggingFace Tokenizers
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    token_ids = encoding.ids
                else:
                    # Transformers tokenizer
                    token_ids = tokenizer.encode(text)
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
            else:
                raise AttributeError("Tokenizer does not have encode method")
            
            print(f"  Tokens: {tokens}")
            print(f"  IDs: {token_ids}")
            
            # Decoding test
            if hasattr(tokenizer, 'decode'):
                if isinstance(tokenizer, Tokenizer):
                    decoded = tokenizer.decode(token_ids)
                else:
                    decoded = tokenizer.decode(token_ids)
                print(f"  Decoded: {decoded}")
                
                # Check round-trip consistency
                if decoded.strip() == text.strip():
                    print("  ✓ Round-trip consistency check passed")
                else:
                    print("  ✗ Round-trip consistency check failed")
                    print(f"    Original: '{text}'")
                    print(f"    Decoded: '{decoded}'")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    return True


def validate_tokenizer_file_structure(tokenizer_path: str):
    """
    Validate file structure
    
    Args:
        tokenizer_path: Tokenizer path
    """
    print("\nFile structure validation:")
    print("-" * 30)
    
    path = Path(tokenizer_path)
    
    if path.is_file():
        print(f"File: {path.name}")
        print(f"Size: {path.stat().st_size} bytes")
        
        if path.suffix == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print("✓ JSON format valid")
                
                # Check required fields
                required_fields = ['model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder']
                for field in required_fields:
                    if field in data:
                        print(f"✓ Contains {field}")
                    else:
                        print(f"✗ Missing {field}")
                        
            except json.JSONDecodeError as e:
                print(f"✗ JSON format error: {e}")
                return False
    
    elif path.is_dir():
        print(f"Directory: {path.name}")
        
        # Check common files
        common_files = [
            'tokenizer.json',
            'tokenizer_config.json', 
            'vocab.txt',
            'special_tokens_map.json'
        ]
        
        for file_name in common_files:
            file_path = path / file_name
            if file_path.exists():
                print(f"✓ Contains {file_name}")
            else:
                print(f"- Does not contain {file_name}")
    
    return True


def main():
    """Main function for validate subcommand"""
    parser = argparse.ArgumentParser(
        description="Validate tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tokenizers-converter validate --tokenizer-path ./tokenizer.json
  tokenizers-converter validate -t ./model --test-texts "Hello World" "Test text"
  tokenizers-converter validate -t ./model --basic-only
        """
    )
    
    parser.add_argument(
        "--tokenizer-path", "-t",
        required=True,
        help="Tokenizer path (file or directory)"
    )
    
    parser.add_argument(
        "--test-texts",
        nargs='+',
        default=["Hello, World!", "This is a test.", "你好，世界！"],
        help="List of test texts"
    )
    
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Only perform basic validation"
    )
    
    parser.add_argument(
        "--skip-file-structure",
        action="store_true",
        help="Skip file structure validation"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer path does not exist: {args.tokenizer_path}")
        sys.exit(1)
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(args.tokenizer_path)
        
        # Basic validation
        if not validate_tokenizer_basic(tokenizer, args.tokenizer_path):
            print("Basic validation failed")
            sys.exit(1)
        
        # Functionality validation
        if not args.basic_only:
            if not validate_tokenizer_functionality(tokenizer, args.test_texts):
                print("Functionality validation failed")
                sys.exit(1)
        
        # File structure validation
        if not args.skip_file_structure:
            if not validate_tokenizer_file_structure(args.tokenizer_path):
                print("File structure validation failed")
                sys.exit(1)
        
        print("\n✓ Validation completed, all checks passed!")
        
    except Exception as e:
        print(f"Error: Exception occurred during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 