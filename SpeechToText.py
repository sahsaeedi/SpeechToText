import librosa
import torch
import torchaudio
import numpy as np
import pickle
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class SpeechToText:

    def __init__(self, voice):
        self.voice = voice
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):

        model = pickle.load( open("./STT_model.pkl", "rb"))
        processor = pickle.load( open("./STT_processor.pkl", "rb"))

        return model, processor

    def speech_processing(self):

        speech_array, sampling_rate = torchaudio.load(self.voice)
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, 16_000)

        return speech_array

    def predict(self):

        model, processor = self.load_model()
        voice = self.speech_processing()

        voice = voice.tolist()
        
        features = processor(voice, sampling_rate=16_000, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits 
        pred_ids = torch.argmax(logits, dim=-1)

        predict_sentence = processor.batch_decode(pred_ids)[0]

        return predict_sentence








