from .talker import SpeechGeneratorSNAC


def build_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type', 'snac_ar')
    if generator_type == 'snac_ar':
        return SpeechGeneratorSNAC(config)

    raise ValueError(f'Unknown generator type: {generator_type}')
