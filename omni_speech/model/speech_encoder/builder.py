from .speech_encoder import HFWhisperWrappedEncoder, WhisperWrappedEncoder, HubertWrappedEncoder


def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if speech_encoder_type == "whisper_hf":
        return HFWhisperWrappedEncoder.load(config)
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)
    if "hubert" in speech_encoder_type.lower():
        return HubertWrappedEncoder.load(config)
        

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
