#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import sys


def main():
    """Main CLI entry function"""
    subcommands = {
        "convert",
        "list", 
        "validate"
    }
    
    if len(sys.argv) < 2:
        print(f"Error: Subcommand required. Available subcommands: {', '.join(sorted(subcommands))}")
        print("\nUsage:")
        print("  tokenizers-converter convert <args>    # Convert tokenizer")
        print("  tokenizers-converter list              # List available converters")
        print("  tokenizers-converter validate <args>   # Validate tokenizer")
        sys.exit(1)
    
    subcommand = sys.argv.pop(1)
    
    if subcommand not in subcommands:
        print(f"Error: Unknown subcommand '{subcommand}'")
        print(f"Available subcommands: {', '.join(sorted(subcommands))}")
        sys.exit(1)
    
    try:
        # Dynamically import subcommand module
        submodule = importlib.import_module(f"tokenizers_converter.commands.{subcommand}")
        
        # Call subcommand main function
        submodule.main()
    except ImportError as e:
        print(f"Error: Cannot import subcommand module '{subcommand}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Exception occurred while executing subcommand: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
