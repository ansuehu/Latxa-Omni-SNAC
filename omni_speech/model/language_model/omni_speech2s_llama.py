import inspect
from typing import List, Optional, Tuple, Union

import torch

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from omni_speech.model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM
from omni_speech.model.speech_generator.builder import build_speech_generator
from omni_speech.model.speech_generator.talker import DEFAULT_TALKER_CONFIG


TALKER_CONFIG_KEYS = (
    'speech_generator_type', 'talker_config', 'talker_layout', 'talker_read',
    'talker_write', 'talker_loss_weight', 'talker_attn_implementation',
    'tune_speech_generator_only',
)


class OmniSpeech2SConfig(LlamaConfig):
    model_type = "omni_speech2s_llama"


class OmniSpeech2SLlamaForCausalLM(OmniSpeechLlamaForCausalLM):
    config_class = OmniSpeech2SConfig

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()
        if getattr(config, "speech_generator_type", None) == "snac_ar":
            self.speech_generator = build_speech_generator(config)

    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = 'snac_ar'
        self.config.talker_config = getattr(model_args, 'talker_config', None) or dict(DEFAULT_TALKER_CONFIG)
        self.config.talker_layout = getattr(model_args, 'talker_layout', 'interleave')
        self.config.talker_read = getattr(model_args, 'talker_read', 3)
        self.config.talker_write = getattr(model_args, 'talker_write', 28)
        self.config.talker_loss_weight = getattr(model_args, 'talker_loss_weight', 1.0)
        self.config.talker_attn_implementation = getattr(model_args, 'talker_attn_implementation', 'sdpa')
        self.config.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', True)
        if getattr(self, "speech_generator", None) is None:
            self.speech_generator = build_speech_generator(self.config)
            self.speech_generator.to(dtype=self.dtype, device=self.device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if tgt_units is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, speech=speech,
                speech_lengths=speech_lengths, return_dict=return_dict,
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
            llama_output = LlamaForCausalLM.forward(
                self,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            talker_loss = self.speech_generator(
                llama_output.hidden_states[-1], labels, text_embeds, tgt_units
            )
            loss = llama_output.loss + talker_loss * self.config.talker_loss_weight
            logits = llama_output.logits

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
        """Text answer, then SNAC tokens for it. Returns (text ids, flat SNAC tokens).

        Batch size 1. The talker is conditioned on the hidden states of a single
        forward pass over prompt + answer, exactly as in training.
        """
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
        answer_ids = LlamaForCausalLM.generate(self, inputs_embeds=prompt_embeds, **kwargs)
        answer_embeds = self.get_model().embed_tokens(answer_ids)
        hidden = self.get_model()(
            inputs_embeds=torch.cat([prompt_embeds, answer_embeds], dim=1), use_cache=False
        ).last_hidden_state[:, -answer_ids.shape[1]:]
        cond = self.speech_generator.fuse(hidden[0], answer_embeds[0])
        units = self.speech_generator.generate_units(cond, **(talker_kwargs or {}))
        return answer_ids, units

    @torch.no_grad()
    def stream_generate(
        self,
        inputs: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        speech_embeds: Optional[list] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id=None,
        talker_kwargs: Optional[dict] = None,
    ):
        """Yield ("text", [token ids]) and ("units", [flat SNAC tokens]) as they are made.

        Batch size 1. The LLM decodes one token at a time with a KV cache. Feeding
        answer token t back in gives its last hidden state, which is exactly the
        state the talker saw for t in training (causal attention), so each token's
        condition goes to the talker one step after it is sampled. The talker pulls
        conditions as it needs them (``read`` per ``write`` audio tokens), so audio
        starts after a few text tokens.
        """
        from transformers import DynamicCache

        if inputs.shape[0] != 1:
            raise ValueError("speech generation supports batch size 1")
        (
            _, _, _, _, prompt_embeds, _
        ) = self.prepare_inputs_labels_for_speech_and_text(
            inputs, None, None, None, None, speech, speech_lengths, speech_embeds=speech_embeds,
        )
        model = self.get_model()
        if prompt_embeds is None:
            prompt_embeds = model.embed_tokens(inputs)
        if eos_token_id is None:
            eos_token_id = self.generation_config.eos_token_id
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        text_ids = []

        def next_token(hidden):
            logits = self.lm_head(hidden).float()
            if temperature <= 0:
                return int(logits.argmax())
            probs = torch.softmax(logits / temperature, -1)
            if top_p < 1.0:
                sorted_probs, order = torch.sort(probs, descending=True)
                sorted_probs[sorted_probs.cumsum(-1) - sorted_probs > top_p] = 0
                probs = torch.zeros_like(probs).scatter(0, order, sorted_probs)
            return int(torch.multinomial(probs, 1))

        def conditions():
            # transformers 5 takes the config, which Qwen3.5 needs for its linear-attention state.
            if "config" in inspect.signature(DynamicCache.__init__).parameters:
                cache = DynamicCache(config=self.config)
            else:
                cache = DynamicCache()
            out = model(inputs_embeds=prompt_embeds, past_key_values=cache, use_cache=True)
            for _ in range(max_new_tokens):
                token = next_token(out.last_hidden_state[0, -1])
                text_ids.append(token)
                embeds = model.embed_tokens(torch.tensor([[token]], device=prompt_embeds.device))
                out = model(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
                yield self.speech_generator.fuse(out.last_hidden_state[0], embeds[0])[0]
                if token in eos:
                    return

        sent_text, frame = 0, []
        for unit in self.speech_generator.stream_units(conditions(), **(talker_kwargs or {})):
            if len(text_ids) > sent_text:
                yield "text", text_ids[sent_text:]
                sent_text = len(text_ids)
            frame.append(unit)
            if len(frame) == 7:
                yield "units", frame
                frame = []
        if len(text_ids) > sent_text:
            yield "text", text_ids[sent_text:]


AutoConfig.register("omni_speech2s_llama", OmniSpeech2SConfig)
AutoModelForCausalLM.register(OmniSpeech2SConfig, OmniSpeech2SLlamaForCausalLM)
