# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        import whisper
        encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        replace_layer_norm(encoder)
        
        return encoder


class HFWhisperWrappedEncoder(nn.Module):
    """Use the Hugging Face Whisper encoder when the OpenAI .pt is unavailable."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    @classmethod
    def load(cls, model_config):
        from transformers import WhisperModel

        model = WhisperModel.from_pretrained(model_config.speech_encoder)
        return cls(model.encoder)

    def forward(self, input_features):
        return self.encoder(input_features=input_features).last_hidden_state

class HubertWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        from transformers import HubertModel
        encoder = HubertModel.from_pretrained(model_config.speech_encoder)
        
        return encoder
