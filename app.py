
import torch
import base64
import io
import torch
import numpy as np
import time, os
from scipy.io.wavfile import read
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

MODEL_NAME = 'laudite-ai/wav2vec2-large-a3500-e30'
API_KEY = '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'

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init(hotwords=[], use_lm_if_possible = True):
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

def buffer_to_text(audio_buffer):
    global model, processor, device, use_lm_if_possible
    start = time.perf_counter()
    
    if len(audio_buffer) == 0:
        return ''

    inputs = processor(
        torch.tensor(audio_buffer, device=device),
        sampling_rate=16_000,
        return_tensors='pt',
        padding=True).to(device)
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    if hasattr(processor, 'decoder') and use_lm_if_possible:
        transcription = processor.decode(
            logits[0].cpu().numpy(),
            hotwords=hotwords,
            output_word_offsets=True)

        transcription = transcription.text

    else:
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    end = time.perf_counter()
    inference_time = end - start
    return transcription, inference_time

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(request:dict) -> dict:
    global model
    audio_file = request.get('audio', None)
    decoded_data = base64.b64decode(str(audio_file))
    rate, data = read(io.BytesIO(decoded_data))
    data = data / (2**15)
    data_array = np.array(data)

    with open("output.txt", "w") as f:
        f.write(str(data_array))

    try:
        text,inference_time = buffer_to_text(data_array)
    except Exception as e:
        print(e)
        text = ''
        inference_time = 0

    result = {
        'text': text,
        'inference_time': inference_time
    }

    print('Resultado: ', result)

    return result