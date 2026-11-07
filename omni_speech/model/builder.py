# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from omni_speech.model import *
from omni_speech.model.speech_encoder.builder import build_speech_encoder
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Stage 2 (frozen LLM) checkpoints hold only the talker; their config's
# talker_base_model names the Stage 1 checkpoint with everything else.
TALKER_WEIGHTS_NAME = "talker.safetensors"


def talker_only_base(model_path):
    """Base checkpoint of a talker-only checkpoint, or None for a full one."""
    if not os.path.isfile(os.path.join(model_path, TALKER_WEIGHTS_NAME)):
        return None
    return AutoConfig.from_pretrained(model_path).talker_base_model


def load_talker_weights(model, model_path):
    from safetensors.torch import load_file

    state = load_file(os.path.join(model_path, TALKER_WEIGHTS_NAME))
    prefix = "speech_generator."
    model.speech_generator.load_state_dict({k[len(prefix):]: v for k, v in state.items()})


QWEN3_5_MODEL_TYPES = ("qwen3_5_text", "omni_speech_qwen3_5", "omni_speech2s_qwen3_5")


def is_qwen3_5(config):
    return getattr(config, "model_type", None) in QWEN3_5_MODEL_TYPES


def load_pretrained_model(model_path, model_base, is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM
    if model_base is None and not is_lora and is_qwen3_5(AutoConfig.from_pretrained(model_path)):
        # Trained in bf16; the gated delta rule layers are not tested in fp16.
        if s2s:
            from omni_speech.model.language_model.omni_speech2s_qwen3_5 import OmniSpeech2SQwen3_5ForCausalLM
            model_cls = OmniSpeech2SQwen3_5ForCausalLM
        else:
            from omni_speech.model.language_model.omni_speech_qwen3_5 import OmniSpeechQwen3_5ForCausalLM
            model_cls = OmniSpeechQwen3_5ForCausalLM
        kwargs['torch_dtype'] = torch.bfloat16
        kwargs.setdefault('attn_implementation', 'sdpa')

    # Load OmniSpeech model
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        non_lora_trainables_path = os.path.join(model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_trainables_path):
            non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading OmniSpeech from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)
        
        speech_projector_weights = torch.load(os.path.join(model_path, 'speech_projector.bin'), map_location='cpu')
        speech_projector_weights = {k: v.to(torch.float16) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        base = talker_only_base(model_path)
        if base is not None:
            kwargs['config'] = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(
            base or model_path,
            low_cpu_mem_usage=False,
            **kwargs
        )
        if base is not None:
            load_talker_weights(model, model_path)

    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(dtype=model.dtype)
    model = model.to(device=device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len

def create_model(model_path, model_base, is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM
    if is_qwen3_5(kwargs.get('config') or AutoConfig.from_pretrained(model_path)):
        if s2s:
            from omni_speech.model.language_model.omni_speech2s_qwen3_5 import OmniSpeech2SQwen3_5ForCausalLM
            model_cls = OmniSpeech2SQwen3_5ForCausalLM
        else:
            from omni_speech.model.language_model.omni_speech_qwen3_5 import OmniSpeechQwen3_5ForCausalLM
            model_cls = OmniSpeechQwen3_5ForCausalLM
        # Only every fourth layer is full attention; sdpa avoids building flash-attn for torch 2.8.
        kwargs.setdefault('attn_implementation', 'sdpa')
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        # if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        #     non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        #     non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        # if any(k.startswith('model.model.') for k in non_lora_trainables):
        #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        #model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
        
    else:
        if model_base is not None:
            # Unlike load_pretrained_model, this branch loads everything
            # (weights and tokenizer) from model_path alone; model_base is
            # not consulted. Warn instead of silently ignoring it, since
            # both training SLURM scripts pass --model-base here.
            print(f"WARNING: model_base={model_base!r} is ignored by create_model() "
                  f"when is_lora=False; loading tokenizer and weights from model_path={model_path!r} only.")
        tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side="right",padding=True, use_fast=True)
        kwargs.setdefault('attn_implementation', 'flash_attention_2')
        base = talker_only_base(model_path)
        if base is not None:
            # The base (Stage 1) config has no talker; the Stage 2 one does.
            kwargs.setdefault('config', AutoConfig.from_pretrained(model_path))
        model = model_cls.from_pretrained(
            base or model_path,
            low_cpu_mem_usage=False,
            **kwargs,
        )
        if base is not None:
            load_talker_weights(model, model_path)
    
    if s2s:
        model.initialize_speech_generator(model.config)
    # model = model.to(device=device,dtype=torch.bfloat16)
    if model.get_model().get_speech_encoder() is None:
        # Built outside from_pretrained (always for Qwen3.5), so it needs the model's dtype.
        model.get_model().speech_encoder = build_speech_encoder(model.config).to(dtype=model.dtype)
    # model.get_model().speech_encoder.to(device=device, dtype=torch.bfloat16)
    #冻住speech_encoder的参数
    for param in model.get_model().speech_encoder.parameters():
        param.requires_grad = False
        
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
