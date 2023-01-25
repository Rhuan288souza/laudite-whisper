# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

MODEL_NAME = 'laudite-ai/wav2vec2-large-a3500-e30'
API_KEY = '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'

def download_model():
    global model,processor,device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('#laudite-sr: dispositivo:\n', device)
    
    if use_lm_if_possible:
        processor = AutoProcessor.from_pretrained(MODEL_NAME,use_auth_token=API_KEY)
    else:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME,use_auth_token=API_KEY)

    model = AutoModelForCTC.from_pretrained(MODEL_NAME,use_auth_token=API_KEY)
    print('#laudite-sr: modelo carregado.')
    hotwords = hotwords
    use_lm_if_possible = use_lm_if_possible

    model.to(device)
    
if __name__ == "__main__":
    download_model()