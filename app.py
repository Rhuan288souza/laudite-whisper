
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
    global model, processor, device
    start = time.perf_counter()
    
    if len(audio_buffer) == 0:
        return ''
    
    with torch.no_grad():
        inputs = torch.tensor(
            processor.feature_extractor(
                audio_buffer, sampling_rate=16_000
            ).input_features[0],
            device=device).unsqueeze(0)

    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    inference_time = time.perf_counter()-start
    return transcription, inference_time

def transcribe_audio(audio, is_wav_file=False):
    audio_buffer = np.frombuffer(audio, dtype=np.int16) / 32767

    start = time.perf_counter()
    try:
        transcription = buffer_to_text(audio_buffer)
        transcription = transcription.lower()
        inference_time = time.perf_counter()-start
        if transcription != "":
            print('transcription: ', transcription)
            print('inference_time: ', inference_time)
            
        return transcription, inference_time

    except Exception as e:
        print('inference error')
        print(e)

        return None, None

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(request:dict) -> dict:
    global model
    audio_file = request.get('audio', None)
    decoded_data = base64.b64decode(str(audio_file))
    rate, data = read(io.BytesIO(decoded_data))
    data = data / (2**15)
    data_array = np.array(data)

    try:
        text,inference_time = transcribe_audio(data_array)
    except:
        text = ''
        inference_time = 0

    result = {
        'text': text,
        'inference_time': inference_time
    }

    print('Resultado: ', result)

    return result