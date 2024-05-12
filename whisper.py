from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", cache_dir=os.path.join(current_dir, 'data'))

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=os.path.join(current_dir, 'model'))
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
        # predicted_ids = model.generate(input_features)[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

result = librispeech_test_clean.map(map_to_pred)

wer = load("wer")
print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))
