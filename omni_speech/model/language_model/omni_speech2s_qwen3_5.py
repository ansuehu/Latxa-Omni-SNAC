"""Latxa-Qwen3.5 thinker with the SNAC talker (Stage 2), for transformers >= 5.2 (the omni-qwen env).

Same as OmniSpeech2SLlamaForCausalLM, with Qwen3_5ForCausalLM as the language model. The talker's
input projections take the thinker's hidden size from the config (2560 for the 4B).
"""

from typing import Optional

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from omni_speech.model.language_model.omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM
from omni_speech.model.language_model.omni_speech_qwen3_5 import (
    OmniSpeechQwen3_5Config, OmniSpeechQwen3_5ForCausalLM,
)
from omni_speech.model.speech_generator.builder import build_speech_generator


class OmniSpeech2SQwen3_5Config(OmniSpeechQwen3_5Config):
    model_type = "omni_speech2s_qwen3_5"


class OmniSpeech2SQwen3_5ForCausalLM(OmniSpeechQwen3_5ForCausalLM):
    config_class = OmniSpeech2SQwen3_5Config

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "speech_generator_type", None) == "snac_ar":
            self.speech_generator = build_speech_generator(config)
        self.post_init()

    # Model-independent: the talker settings and the streaming decoding loop.
    initialize_speech_generator = OmniSpeech2SLlamaForCausalLM.initialize_speech_generator
    stream_generate = OmniSpeech2SLlamaForCausalLM.stream_generate

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
        tgt_units: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if tgt_units is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                speech=speech, speech_lengths=speech_lengths, **kwargs,
            )

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
        if inputs_embeds is None:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        # Answer token ids are the labels themselves, so the talker's text
        # embeddings come from the (frozen in Stage 2) LLM embedding table.
        text_embeds = self.get_model().embed_tokens(labels.clamp(min=0))

        if getattr(self.config, 'tune_speech_generator_only', True):
            with torch.no_grad():
                hidden = self.get_model()(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).last_hidden_state
            loss = self.speech_generator(hidden, labels, text_embeds.detach(), tgt_units)
            logits = None
        else:
            llm_output = Qwen3_5ForCausalLM.forward(
                self,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
            )
            talker_loss = self.speech_generator(
                llm_output.hidden_states[-1], labels, text_embeds, tgt_units
            )
            loss = llm_output.loss + talker_loss * self.config.talker_loss_weight
            logits = llm_output.logits

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        speech_embeds: Optional[torch.Tensor] = None,
        talker_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Text answer, then SNAC tokens for it. Returns (text ids, flat SNAC tokens). Batch size 1."""
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if inputs.shape[0] != 1:
            raise ValueError("speech generation supports batch size 1")
        (
            _, _, _, _, prompt_embeds, _
        ) = self.prepare_inputs_labels_for_speech_and_text(
            inputs, position_ids, attention_mask, None, None,
            speech, speech_lengths, speech_embeds=speech_embeds,
        )
        if prompt_embeds is None:
            prompt_embeds = self.get_model().embed_tokens(inputs)
        answer_ids = Qwen3_5ForCausalLM.generate(self, inputs_embeds=prompt_embeds, **kwargs)
        answer_embeds = self.get_model().embed_tokens(answer_ids)
        hidden = self.get_model()(
            inputs_embeds=torch.cat([prompt_embeds, answer_embeds], dim=1), use_cache=False
        ).last_hidden_state[:, -answer_ids.shape[1]:]
        cond = self.speech_generator.fuse(hidden[0], answer_embeds[0])
        units = self.speech_generator.generate_units(cond, **(talker_kwargs or {}))
        return answer_ids, units


# Same weight renaming as OmniSpeechQwen3_5ForCausalLM (see there): registered by class name.
register_checkpoint_conversion_mapping(
    "OmniSpeech2SQwen3_5ForCausalLM", get_checkpoint_conversion_mapping("qwen3_5_text"), overwrite=True,
)
AutoConfig.register("omni_speech2s_qwen3_5", OmniSpeech2SQwen3_5Config)
AutoModelForCausalLM.register(OmniSpeech2SQwen3_5Config, OmniSpeech2SQwen3_5ForCausalLM)
