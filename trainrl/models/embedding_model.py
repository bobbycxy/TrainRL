'''
Code that will load the embedding layer of the model. 

This will be used to generate embeddings for the input data.
'''
import torch
from models.components.tokenizer import build_tokenizer
from models.utils import build_huggingface_model

class EmbedderInterface(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.eot_token_id = ... # ... is a placeholder for the end of text token
    
    def forward(self, token_ids):
        """
        Forward pass of the model
        """
        raise NotImplementedError
    
    def tokenize_input(self, input_string):
        """
        Tokenize the input string
        """
        raise NotImplementedError
    
    def decode(self, token_ids):
        """
        Decode the token ids
        """
        raise NotImplementedError
    
    def inference(self, input_str, add_eot=False):
        """
        Perform inference
        """
        raise NotImplementedError
    
    def pad_batch(self, token_lists):
        """
        Pad the batch of token lists
        """
        raise NotImplementedError
    

class HuggingFaceEmbedder(EmbedderInterface):

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        self.embeddings = build_huggingface_model(model_cfg).get_input_embeddings()
        self.tokenizer = build_tokenizer(model_cfg)
        self.eot_token_id = self.tokenizer.eot_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, token_ids):
        return self.embeddings(token_ids)
    
    def tokenize_input(self, input_string, add_eot=True, truncate_from=None):
        token_ids = self.tokenizer.encode(input_string)
        
        if add_eot:
            token_ids.append(self.eot_token_id)

        if truncate_from != None:
            token_ids = self.truncate(
                token_lists=[token_ids],
                truncate_from=truncate_from
            )[0]

        return token_ids

    def truncate(self, token_lists, truncate_from='right'):
        max_length = self.model_cfg.get("context_window", 1024)
        if truncate_from == 'right':
            return [token_list[:max_length] for token_list in token_lists]
        elif truncate_from == 'left':
            return [token_list[-max_length:] for token_list in token_lists]
        else:
            raise ValueError("Invalid direction")
    
    def decode(self, token_ids):
        return self.tokenizer.decode_batch(token_ids)

    def pad_batch(self, token_lists, direction='right'):
        return self.tokenizer.pad_batch(token_lists, direction=direction)


