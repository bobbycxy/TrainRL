'''
The tokenizer is meant to help encode the given input text into token ids that the model can understand.
'''

from transformers import AutoTokenizer

class TokenizerInterface:
    def __init__(self):
        pass

    def encode(self, input_string):
        """
        Encode the input string
        """
        raise NotImplementedError
    
    def encode_batch(self, input_strings):
        """
        Encode the batch of input strings
        """
        raise NotImplementedError
    
    def decode(self, token_ids):
        """
        Decode the token ids
        """
        raise NotImplementedError
    
    def decode_batch(self, token_id_lists):
        """
        Decode the batch of token ids
        """
        raise NotImplementedError
    
    def pad_batch(self, token_lists, direction='right'):
        """
        Pad the batch of token lists
        """
        max_len = max([len(token_list) for token_list in token_lists])
        padded_token_lists = []
        mask_lists = []
        for token_list in token_lists:
            if direction == 'right':
                padded_token_list = token_list + [self.pad_token_id] * (max_len - len(token_list))
                mask_list = [1] * len(token_list) + [0] * (max_len - len(token_list))
            elif direction == 'left':
                padded_token_list = [self.pad_token_id] * (max_len - len(token_list)) + token_list
                mask_list = [0] * (max_len - len(token_list)) + [1] * len(token_list)
            else:
                raise ValueError("Invalid direction")
            padded_token_lists.append(padded_token_list)
            mask_lists.append(mask_list)

        return padded_token_lists, mask_lists
    
class HuggingFaceTokenizer(TokenizerInterface):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eot_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, input_string):
        """
        Encode the input string, without any attention mask
        """
        return self.tokenizer.encode(input_string)
    
    def encode_batch(self, input_strings):
        """
        Encode the batch of input strings, without any attention mask
        """
        return self.tokenizer.batch_encode_plus(input_strings, add_special_tokens=False, padding=False)['input_ids']
    
    def decode(self, token_ids):
        """
        Decode the token ids
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def decode_batch(self, token_id_lists):
        """
        Decode the batch of token ids
        """
        return self.tokenizer.batch_decode(token_id_lists, skip_special_tokens=True)


TOKENIZER_DICT = {
    "huggingface": lambda model_cfg: HuggingFaceTokenizer(model_cfg),
    "llama3.2-1B": lambda model_cfg: HuggingFaceTokenizer(model_cfg),
}

def build_tokenizer(cfg):
    """
    Build the tokenizer
    """
    return TOKENIZER_DICT[cfg['tokenizer_type']](cfg) 