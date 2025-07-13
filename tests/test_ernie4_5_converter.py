import os
import random
from functools import cached_property
from unittest import TestCase
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenizers_converter.tokenizers.ernie4_5_converter import Ernie45Converter


class TestErnie45Converter(TestCase):
    """Test ERNIE 4.5 tokenizer converter functionality"""
    
    MODEL_NAME = "mlx-community/ERNIE-4.5-0.3B-PT-4bit"
    LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models", MODEL_NAME)
    
    # Test string collection
    TEST_STRINGS = [
        " {\n",
        "  {\n", 
        "x  {\n",
        "----------------------------------------------------------------------------\n",
        "\n \n",
        "\n  \n",
        '// -----------------------------------------------------------------------\n',
        '-----------------------------------------------------------------------\n',
        "Hello world!",
        "你好世界",
        "مختلف، أنواع مختلفة تماما",
        "Bonjour le monde",
        "こんにちは世界",
        "",  # Empty string
        " ",  # Single space
        "\t",  # Tab
        "\n",  # Newline
    ]

    def setUp(self):
        """Set up test environment"""
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.LOCAL_DIR), exist_ok=True)
        
        # Download model files
        snapshot_download(
            self.MODEL_NAME,
            local_dir=self.LOCAL_DIR,
            allow_patterns=[
                "added_tokens.json",
                "special_tokens_map.json",
                "tokenization_ernie4_5.py",
                "tokenizer.model",
                "tokenizer_config.json",
            ]
        )
        
        # Initialize error tracking
        self._error_count = 0
        self._error_lock = Lock()

    @cached_property
    def original_tokenizer(self):
        """Get original tokenizer (cached property to avoid repeated loading)"""
        return AutoTokenizer.from_pretrained(self.LOCAL_DIR, trust_remote_code=True)

    @cached_property
    def converted_tokenizer(self) -> PreTrainedTokenizerFast:
        """Get converted tokenizer (cached property to avoid repeated creation)"""
        converter = Ernie45Converter(self.original_tokenizer)
        return PreTrainedTokenizerFast(
            tokenizer_object=converter.converted(),
            model_input_names=self.original_tokenizer.model_input_names,
            clean_up_tokenization_spaces=False,
            # Pass special tokens from original tokenizer
            bos_token=self.original_tokenizer.bos_token,
            eos_token=self.original_tokenizer.eos_token,
            unk_token=self.original_tokenizer.unk_token,
            sep_token=self.original_tokenizer.sep_token,
            pad_token=self.original_tokenizer.pad_token,
            cls_token=self.original_tokenizer.cls_token,
            mask_token=self.original_tokenizer.mask_token,
            additional_special_tokens=self.original_tokenizer.additional_special_tokens,
        )

    @cached_property
    def xnli_dataset(self):
        """Get XNLI dataset (cached property to avoid repeated loading)"""
        return load_dataset(
            "facebook/xnli",
            data_files={
                "validation": "all_languages/validation-*.parquet"
            },
            split="validation"
        )

    def test_ernie4_5_converter_xnli(self):
        """Test ERNIE 4.5 converter performance on XNLI dataset with multi-threading"""
        text_pairs = [
            (lang, text) 
            for premise in self.xnli_dataset["premise"]
            for lang, text in premise.items()
            if text
        ]
        
        success_count = 0
        max_workers = min(32, len(text_pairs))
        
        def process_text_pair(lang_text_pair):
            """Process a single text pair"""
            lang, text = lang_text_pair
            try:
                self._verify_tokenization(lang, text)
                return True, lang, None
            except AssertionError as e:
                with self._error_lock:
                    self._error_count += 1
                    if self._error_count <= 5:
                        print(f"Error in {lang}: {e}")
                return False, lang, e
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_text_pair, pair): pair for pair in text_pairs}
            
            # Track completion with progress bar
            with tqdm(total=len(text_pairs), desc="Testing XNLI dataset (multi-threaded)") as pbar:
                for future in as_completed(futures):
                    success, lang, error = future.result()
                    if success:
                        success_count += 1
                    
                    pbar.update(1)
                    
                    # Terminate early if too many errors
                    if self._error_count > 50:
                        # Cancel remaining tasks
                        for remaining_future in futures:
                            remaining_future.cancel()
                        break
        
        total_count = len(text_pairs)
        print(f"XNLI test completed: {success_count}/{total_count} successful, {self._error_count} errors")
        
        if self._error_count > total_count * 0.1:
            self.fail(f"Too many errors: {self._error_count}/{total_count} ({self._error_count/total_count*100:.1f}%)")

    def test_ernie4_5_converter_xnli_sample(self):
        """Test ERNIE 4.5 converter on 5 random samples per language"""
        lang_texts = {}
        random_samples = 50
        for premise in self.xnli_dataset["premise"]:
            for lang, text in premise.items():
                if text:
                    lang_texts.setdefault(lang, []).append(text)

        # Calculate total samples for progress bar
        total_samples = sum(min(random_samples, len(texts)) for texts in lang_texts.values())
        
        with tqdm(total=total_samples, desc="Testing XNLI sample") as pbar:
            for lang, texts in lang_texts.items():
                samples = random.sample(texts, min(random_samples, len(texts)))
                for text in samples:
                    self._verify_tokenization(lang, text)
                    pbar.update(1)

    def test_predefined_strings(self):
        """Test predefined string collection"""
        for i, text in enumerate(self.TEST_STRINGS):
            with self.subTest(test_case=i, text=repr(text)):
                self._verify_tokenization(f"test_{i}", text)

    def test_special_tokens_consistency(self):
        """Test special token consistency"""
        special_tokens = [
            self.original_tokenizer.bos_token,
            self.original_tokenizer.eos_token,
            self.original_tokenizer.unk_token,
            self.original_tokenizer.sep_token,
            self.original_tokenizer.pad_token,
            self.original_tokenizer.cls_token,
            self.original_tokenizer.mask_token,
        ]
        
        for token in special_tokens:
            if token:  # Skip None values
                with self.subTest(token=token):
                    self._verify_tokenization("special_token", token)

    def test_empty_and_whitespace(self):
        """Test empty string and whitespace handling"""
        edge_cases = ["", " ", "\t", "\n", "\r", "  ", "\n\n", "\t\t"]
        
        for text in edge_cases:
            with self.subTest(text=repr(text)):
                self._verify_tokenization("whitespace", text)

    def _verify_tokenization(self, context: str, text: str):
        """Verify tokenization result consistency
        
        Args:
            context: Test context (for error messages)
            text: Text to test
        """
        # Test encoding with special tokens
        original_ids = self.original_tokenizer.encode(text)
        converted_ids = self.converted_tokenizer.encode(text)
        
        self.assertEqual(
            original_ids,
            converted_ids,
            f"Token ID mismatch (context: {context})\n"
            f"Text: {repr(text)}\n"
            f"Original: {original_ids}\n"
            f"Converted: {converted_ids}"
        )

        # Test encoding without special tokens
        original_ids_no_special = self.original_tokenizer.encode(text, add_special_tokens=False)
        converted_ids_no_special = self.converted_tokenizer.encode(text, add_special_tokens=False)
        
        self.assertEqual(
            original_ids_no_special,
            converted_ids_no_special,
            f"Token ID mismatch (no special tokens, context: {context})\n"
            f"Text: {repr(text)}\n"
            f"Original: {original_ids_no_special}\n"
            f"Converted: {converted_ids_no_special}"
        )

        # Test decoding consistency
        original_decoded = self.original_tokenizer.decode(original_ids, skip_special_tokens=True)
        converted_decoded = self.converted_tokenizer.decode(converted_ids, skip_special_tokens=True)
        
        self.assertEqual(
            original_decoded,
            converted_decoded,
            f"Decoding result mismatch (context: {context})\n"
            f"Original text: {repr(text)}\n"
            f"Original decoded: {repr(original_decoded)}\n"
            f"Converted decoded: {repr(converted_decoded)}"
        )

    def test_tokenizer_properties(self):
        """Test basic tokenizer properties"""
        # Vocabulary size should be consistent
        self.assertEqual(
            self.original_tokenizer.vocab_size,
            self.converted_tokenizer.vocab_size,
            "Vocabulary size mismatch"
        )
        
        # Special token IDs should be consistent
        special_token_pairs = [
            ("bos_token_id", self.original_tokenizer.bos_token_id),
            ("eos_token_id", self.original_tokenizer.eos_token_id),
            ("unk_token_id", self.original_tokenizer.unk_token_id),
            ("sep_token_id", self.original_tokenizer.sep_token_id),
            ("pad_token_id", self.original_tokenizer.pad_token_id),
            ("cls_token_id", self.original_tokenizer.cls_token_id),
            ("mask_token_id", self.original_tokenizer.mask_token_id),
        ]
        
        for token_name, expected_id in special_token_pairs:
            if expected_id is not None:
                actual_id = getattr(self.converted_tokenizer, token_name)
                self.assertEqual(
                    expected_id,
                    actual_id,
                    f"{token_name} mismatch: expected {expected_id}, got {actual_id}"
                )

    def _verify_tokenization_safe(self, context: str, text: str) -> bool:
        """Safe version of _verify_tokenization that returns boolean instead of raising"""
        try:
            self._verify_tokenization(context, text)
            return True
        except AssertionError:
            return False

    def test_save(self):
        self.converted_tokenizer.save_pretrained(f"{self.LOCAL_DIR}/converted")