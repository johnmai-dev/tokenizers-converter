# Tokenizer Converter

## Overview

`tokenizers-converter` is a command-line tool for converting and validating tokenizers in different formats.

## Installation

```bash
pip install -e .
```

## Basic Usage

```bash
tokenizers-converter <subcommand> [options]
```

## Available Subcommands

### 1. `convert` - Convert Tokenizer

Convert tokenizer from one format to another.

**Basic Syntax:**
```bash
tokenizers-converter convert -m <model_path> -o <output_path> -t <converter_type>
# python -m tokenizers_converter convert -m mlx-community/ERNIE-4.5-0.3B-PT-4bit -o ./output -t ernie4.5
```

**Arguments:**
- `--model-path`, `-m`: Input model path (required)
- `--output-path`, `-o`: Output path (required)
- `--type`, `-t`: Converter type (required)
- `--vocab-size`: Vocabulary size (optional)

**Supported Converter Types:**
- `ernie4.5` / `ernie-4.5`: ERNIE 4.5 tokenizer converter

**Examples:**
```bash
# Convert ERNIE 4.5 tokenizer
tokenizers-converter convert -m ./ernie-4.5-model -o ./output -t ernie4.5

# Use alias for converter type
tokenizers-converter convert -m ./model -o ./output -t ernie-4.5

# Specify vocabulary size
tokenizers-converter convert -m ./model -o ./output -t ernie4.5 --vocab-size 50000
```

### 2. `list` - List Available Converters

Display all available converters and their information.

**Basic Syntax:**
```bash
tokenizers-converter list [options]
```

**Arguments:**
- `--detailed`, `-d`: Show detailed information (optional)

**Examples:**
```bash
# List all converters
tokenizers-converter list

# Show detailed information
tokenizers-converter list -d
```

**Sample Output:**
```
Available converters:
==================================================
  ernie4.5 - ERNIE 4.5 tokenizer converter

Total 1 converters
Use --detailed flag to view detailed information
```

### 3. `validate` - Validate Tokenizer

Validate tokenizer integrity and functionality.

**Basic Syntax:**
```bash
tokenizers-converter validate -t <tokenizer_path> [options]
```

**Arguments:**
- `--tokenizer-path`, `-t`: Tokenizer path (file or directory) (required)
- `--test-texts`: List of test texts (optional, has default texts)
- `--basic-only`: Only perform basic validation (optional)
- `--skip-file-structure`: Skip file structure validation (optional)

**Examples:**
```bash
# Validate tokenizer.json file
tokenizers-converter validate -t ./tokenizer.json

# Validate model directory
tokenizers-converter validate -t ./model

# Use custom test texts
tokenizers-converter validate -t ./model --test-texts "Hello World" "Test text"

# Only perform basic validation
tokenizers-converter validate -t ./model --basic-only

# Skip file structure validation
tokenizers-converter validate -t ./model --skip-file-structure
```

**Validation Content:**
- Basic validation: tokenizer type, vocabulary size, special tokens
- Functionality validation: encoding/decoding tests, round-trip consistency check
- File structure validation: file format, required fields check