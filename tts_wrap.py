from transformers import AutoProcessor, BarkModel
import scipy
import torch
import nltk;nltk.download('punkt')  # we'll use this to split into sentences
import numpy as np

class TTSWrap:
    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")

        # Move the model to CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.voice_preset = "en_speaker_4"


    def voice(self, text: str):
        sentences = nltk.sent_tokenize(text)
        sample_rate = self.model.generation_config.sample_rate

        silence = np.zeros(int(0.25 * sample_rate))  # 0.25 second of silence

        speech = []

        for sentence in sentences:
            inputs = self.processor(sentence, voice_preset=self.voice_preset)

            ## move alll inputs to CUDA
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            audio_array = self.model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()

            print(f"generated voice for \"{sentence}\"")
            speech += [audio_array]

        result = np.concatenate(speech)
        scipy.io.wavfile.write("output.wav", rate=sample_rate, data=result)
        #result
        return result


