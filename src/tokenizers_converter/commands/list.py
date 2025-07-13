# -*- coding: utf-8 -*-
"""
List available converters command
"""

import argparse
import sys
from tokenizers_converter.tokenizers.ernie4_5_converter import Ernie45Converter


def get_converter_info():
    """Get converter information"""
    return {
        "ernie4.5": {
            "class": Ernie45Converter,
            "description": "ERNIE 4.5 tokenizer converter",
            "input_type": "SentencePiece",
            "output_type": "HuggingFace Tokenizers"
        }
    }


def list_converters(detailed: bool = False):
    """
    List available converters
    
    Args:
        detailed: Whether to show detailed information
    """
    converters = get_converter_info()
    
    if not converters:
        print("No available converters")
        return
    
    print("Available converters:")
    print("=" * 50)
    
    for name, info in converters.items():
        if detailed:
            print(f"\nName: {name}")
            print(f"Description: {info['description']}")
            print(f"Class: {info['class'].__name__}")
            print(f"Input type: {info['input_type']}")
            print(f"Output type: {info['output_type']}")
            print("-" * 30)
        else:
            print(f"  {name} - {info['description']}")
    
    if not detailed:
        print(f"\nTotal {len(converters)} converters")
        print("Use --detailed flag to view detailed information")


def main():
    """Main function for list subcommand"""
    parser = argparse.ArgumentParser(
        description="List available tokenizer converters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tokenizers-converter list                # List all converters
  tokenizers-converter list --detailed     # Show detailed information
        """
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    try:
        list_converters(detailed=args.detailed)
    except Exception as e:
        print(f"Error: Exception occurred while listing converters: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 