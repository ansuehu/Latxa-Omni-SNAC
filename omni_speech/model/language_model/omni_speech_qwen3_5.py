"""Latxa-Qwen3.5 thinker: the Qwen3.5 text model with the Whisper encoder and speech projector.

Same role as OmniSpeechLlamaForCausalLM, for transformers >= 5.2 (the omni-qwen env). The base
checkpoint is the text-only part of HiTZ/Latxa-Qwen3.5-* (extract_qwen3_5_text.py), so weight
names match Qwen3_5ForCausalLM (model.layers.*, tied lm_head) and the vision tower is dropped.
"""

from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextModel

from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM
from ..speech_projector.builder import build_speech_projector


class OmniSpeechQwen3_5Config(Qwen3_5TextConfig):
    model_type = "omni_speech_qwen3_5"


class OmniSpeechQwen3_5Model(OmniSpeechMetaModel, Qwen3_5TextModel):
    config_class = OmniSpeechQwen3_5Config

    def __init__(self, config):
        # transformers 5 builds models on the meta device, where the Whisper encoder cannot be
        # loaded with its own from_pretrained; create_model attaches it after loading instead.
        Qwen3_5TextModel.__init__(self, config)
        if hasattr(config, "speech_encoder"):
            self.speech_projector = build_speech_projector(config)


class OmniSpeechQwen3_5ForCausalLM(Qwen3_5ForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechQwen3_5Config

    def __init__(self, config):
        super(Qwen3_5ForCausalLM, self).__init__(config)
        self.model = OmniSpeechQwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        speech_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None or speech_embeds is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths,
                speech_embeds=speech_embeds
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs


# transformers 5 saves Qwen3.5 text weights under their multimodal names (model.language_model.*)
# and maps them back on load. It skips that mapping for classes defined outside transformers unless
# their class name is registered (the config read from a base checkpoint says "qwen3_5_text").
register_checkpoint_conversion_mapping(
    "OmniSpeechQwen3_5ForCausalLM", get_checkpoint_conversion_mapping("qwen3_5_text"), overwrite=True,
)
AutoConfig.register("omni_speech_qwen3_5", OmniSpeechQwen3_5Config)
AutoModelForCausalLM.register(OmniSpeechQwen3_5Config, OmniSpeechQwen3_5ForCausalLM)
