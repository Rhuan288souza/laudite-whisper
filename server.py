# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration
from sanic import Sanic, response
import subprocess
import app as user_src
import base64
import wave
import io
import torch
import numpy as np
import time, os
from scipy.io.wavfile import read

MODEL_NAME = 'laudite-ai/whisper-base-205h-e30-self-training'
API_KEY = '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'


# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
server = Sanic("my_app")

def init():
    global model,processor,device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('#laudite-sr: dispositivo:\n', device)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
    model.to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='pt', task='transcribe')

def buffer_to_text(audio_buffer):
    global model,processor
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

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/transcribe', methods=["POST"]) 
def inference(request):
    #init()
    audio_file = request.json
    attribute = audio_file['audio']['data']
    decoded_data = base64.b64decode(str(attribute))
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

    return response.json(result)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8000, workers=1)
