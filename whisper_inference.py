import torch
import time, os

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration

#MODEL_NAME = 'laudite-ai/whisper-medium-25h-e30'
MODEL_NAME = 'laudite-ai/whisper-base-205h-e30-self-training'
API_KEY = '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'#os.environ.get('HUGGING_FACE_MODEL_HUB_API_KEY')

class WhisperInference():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('#laudite-sr: dispositivo:\n', self.device)

        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
        self.processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=API_KEY)
        self.model.to(self.device)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language='pt', task='transcribe')

        print('#laudite-sr: modelo carregado.')

    def buffer_to_text(self, audio_buffer):
        start = time.perf_counter()

        if len(audio_buffer) == 0:
            return ''

        with torch.no_grad():
            inputs = torch.tensor(
                self.processor.feature_extractor(
                    audio_buffer, sampling_rate=16_000
                ).input_features[0],
                device=self.device).unsqueeze(0)

        predicted_ids = self.model.generate(inputs)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        inference_time = time.perf_counter()-start
        return transcription, inference_time

whisper_model = WhisperInference()

# import soundfile as sf
# audio_test = '/home/kevyn/Documents/audio.wav'
# audio_buffer, rate = sf.read(audio_test)
# print(whisper_model.buffer_to_text(audio_buffer))

time.sleep(5) # wait 5sec to load the ASR model

