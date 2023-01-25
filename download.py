# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration

MODEL_NAME = 'laudite-ai/whisper-base-205h-e30-self-training'
API_KEY = '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'

def download_model():
    global model,processor,device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('#laudite-sr: dispositivo:\n', device)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
    model.to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='pt', task='transcribe')

if __name__ == "__main__":
    download_model()