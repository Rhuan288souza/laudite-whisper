
import torch
import base64
import io
import torch
import numpy as np
import time, os
from scipy.io.wavfile import read


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        text,inference_time = buffer_to_text(data_array)
    except:
        text = ''
        inference_time = 0

    result = {
        'text': text,
        'inference_time': inference_time
    }

    print('Resultado: ', result)

    return result