from .language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM, OmniSpeechConfig
from .language_model.omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM

try:
    # Qwen3.5 needs transformers >= 5.2 (the omni-qwen env); omni-snac has 4.45.
    from .language_model.omni_speech_qwen3_5 import OmniSpeechQwen3_5ForCausalLM, OmniSpeechQwen3_5Config
    from .language_model.omni_speech2s_qwen3_5 import OmniSpeech2SQwen3_5ForCausalLM, OmniSpeech2SQwen3_5Config
except ImportError:
    pass
