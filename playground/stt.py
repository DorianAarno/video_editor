import torch
from transformers import pipeline

audio = "hi.wav"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
modelTags = "ARTPARK-IISc/whisper-large-v3-vaani-hindi"
transcribe = pipeline(
    task="automatic-speech-recognition",
    model=modelTags,
    device=device,
    return_timestamps="word"  # Enable word-level timestamps
)
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

result = transcribe(audio)
print(result)
print('Transcription: ', result["text"])

# Print word-level timestamps
if "chunks" in result:
    for chunk in result["chunks"]:
        print(f"Word: {chunk['text']}, Start: {chunk['timestamp'][0]}, End: {chunk['timestamp'][1]}")

with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
