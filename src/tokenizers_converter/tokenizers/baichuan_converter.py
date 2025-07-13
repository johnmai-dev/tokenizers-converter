from tokenizers import processors, decoders, Tokenizer, normalizers, pre_tokenizers, AddedToken
from tokenizers.models import BPE
from transformers.convert_slow_tokenizer import SpmConverter, _get_prepend_scheme


class BaichuanConverter(SpmConverter):
    handle_byte_fallback = True
    
    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = False

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer
    
    def vocab(self, proto):
        """Build vocabulary from sentencepiece proto"""
        # Baichuan tokenizer remaps the first few tokens
        # We need to build vocab that matches the original tokenizer's mapping
        vocab = []
        
        # Add tokens in the order they appear in the original tokenizer
        for i in range(len(proto.pieces)):
            original_token = self.original_tokenizer.convert_ids_to_tokens(i)
            # Use the original tokenizer's token, but get score from proto
            if i < len(proto.pieces):
                # For the first few tokens, use score 0.0 as they are special
                if i < 4:  # <pad>, <s>, </s>, <unk>
                    vocab.append((original_token, 0.0))
                else:
                    # For other tokens, use the score from proto
                    # Find the corresponding piece in proto
                    proto_piece = proto.pieces[i]
                    vocab.append((original_token, proto_piece.score))
            else:
                vocab.append((original_token, 0.0))
        
        return vocab

    def unk_id(self, proto):
        """Return UNK token ID"""
        return self.original_tokenizer.unk_token_id

    def tokenizer(self, proto):
        """Create the core tokenizer with BPE model"""
        vocab_scores = self.vocab(proto)
        _, merges = self.SpmExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
        bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
        
        tokenizer = Tokenizer(
            BPE(
                bpe_vocab,
                merges,
                unk_token=proto.trainer_spec.unk_piece,
                fuse_unk=True,
                byte_fallback=self.handle_byte_fallback,
                dropout=None,
            )
        )

        # Handle special tokens and user defined symbols
        # Control tokens are special (type == 3)
        # User defined symbols are not special (type == 4)
        smp_added_tokens = [
            (id, p.piece, p.type == 3 or p.piece in self.special_tokens)
            for id, p in enumerate(proto.pieces)
            if p.type in [3, 4] or p.piece in self.special_tokens
        ]

        # Reproduce weird behaviour in original tokenizer
        # Only add tokens that did not originally exist as single tokens
        bad_added_tokens = set()
        for _, token, _ in smp_added_tokens:
            encoded = self.original_tokenizer.encode(token)
            if len(encoded) != 1:
                bad_added_tokens.add(token)

        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=True, special=special)
                for id, token, special in sorted(smp_added_tokens, key=lambda x: x[0])
                if token not in bad_added_tokens
            ]
        )

        return tokenizer

    def normalizer(self, proto):
        """Replace spaces with ▁ (sentencepiece style)"""
        return normalizers.Replace(pattern=" ", content="▁")

    def pre_tokenizer(self, replacement, add_prefix_space):
        """Baichuan doesn't use pre-tokenization"""
        return None
    
    def decoder(self, replacement, add_prefix_space):
        """Decode tokens back to text"""
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        return decoders.Sequence(sequence)
    
    def post_processor(self):
        """Baichuan doesn't use post-processing"""
        return None 

    def __init__(self, *args):
        super().__init__(*args)
        
        # Set special tokens from original tokenizer
        self.special_tokens = {
            self.original_tokenizer.bos_token,
            self.original_tokenizer.eos_token,
            self.original_tokenizer.unk_token,
            self.original_tokenizer.pad_token,
        }
        
        # Add additional special tokens
        if self.original_tokenizer.additional_special_tokens:
            self.special_tokens.update(self.original_tokenizer.additional_special_tokens)
        
        # Remove None values
        self.special_tokens = {token for token in self.special_tokens if token is not None} 