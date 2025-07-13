# -*- coding: utf-8 -*-
"""
CLI tool tests
"""

import subprocess
import sys
import pytest
from pathlib import Path


def run_cli_command(args):
    """Run CLI command and return result"""
    cmd = [sys.executable, "-m", "tokenizers_converter"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    return result


def test_cli_no_args():
    """Test CLI invocation without arguments"""
    result = run_cli_command([])
    assert result.returncode == 1
    assert "Subcommand required" in result.stdout
    assert "convert" in result.stdout
    assert "list" in result.stdout
    assert "validate" in result.stdout


def test_cli_unknown_command():
    """Test unknown subcommand"""
    result = run_cli_command(["unknown"])
    assert result.returncode == 1
    assert "Unknown subcommand" in result.stdout


def test_list_command():
    """Test list subcommand"""
    result = run_cli_command(["list"])
    assert result.returncode == 0
    assert "Available converters" in result.stdout
    assert "ernie4.5" in result.stdout


def test_list_detailed_command():
    """Test list --detailed subcommand"""
    result = run_cli_command(["list", "--detailed"])
    assert result.returncode == 0
    assert "Available converters" in result.stdout
    assert "ERNIE 4.5 tokenizer converter" in result.stdout
    assert "Ernie45Converter" in result.stdout


def test_convert_help():
    """Test convert --help"""
    result = run_cli_command(["convert", "--help"])
    assert result.returncode == 0
    assert "Convert tokenizer format" in result.stdout
    assert "--pretrained_model_name_or_path" in result.stdout
    assert "--output-path" in result.stdout
    assert "--type" in result.stdout


def test_validate_help():
    """Test validate --help"""
    result = run_cli_command(["validate", "--help"])
    assert result.returncode == 0
    assert "Validate tokenizer" in result.stdout
    assert "--tokenizer-path" in result.stdout
    assert "--test-texts" in result.stdout


def test_convert_missing_args():
    """Test convert with missing required arguments"""
    result = run_cli_command(["convert"])
    assert result.returncode == 2  # argparse error code
    assert "required" in result.stderr or "required" in result.stdout


def test_validate_missing_args():
    """Test validate with missing required arguments"""
    result = run_cli_command(["validate"])
    assert result.returncode == 2  # argparse error code
    assert "required" in result.stderr or "required" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__]) 