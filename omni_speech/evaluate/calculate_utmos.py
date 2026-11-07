import torch
import whisper
import sys
import glob

scores = []
audios = glob.glob(f'{sys.argv[1]}*.wav')
print(audios)
for audio in audios: 
    # print(audio)
    wave = whisper.load_audio(audio)

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    score = predictor(torch.from_numpy(wave).unsqueeze(0), 16000)

    scores.append(score)

print(sum(scores)/len(scores))
