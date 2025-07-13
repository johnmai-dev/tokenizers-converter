from tokenizers import processors, decoders, Tokenizer, normalizers, pre_tokenizers
from transformers.convert_slow_tokenizer import SpmConverter, _get_prepend_scheme


class Ernie45Converter(SpmConverter):
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
    
    def normalizer(self, proto):
        return None

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme, split=False)
    
    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)
    
    def post_processor(self):
        bos_token = self.original_tokenizer.bos_token  # "<s>"
        cls_token = self.original_tokenizer.cls_token  # "<|begin_of_sentence|>"
        sep_token = self.original_tokenizer.sep_token  # "<|end_of_sentence|>"
        
        bos_token_id = self.original_tokenizer.bos_token_id  # 1
        cls_token_id = self.original_tokenizer.cls_token_id  # 100273
        sep_token_id = self.original_tokenizer.sep_token_id  # 100272
        
        return processors.TemplateProcessing(
            single=f"{bos_token} {cls_token} $A {sep_token}",
            pair=f"{bos_token} {cls_token} $A {sep_token} $B {sep_token}",
            special_tokens=[
                (bos_token, bos_token_id),
                (cls_token, cls_token_id),
                (sep_token, sep_token_id),
            ],
        )
