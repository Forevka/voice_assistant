from transformers import AutoProcessor, BarkModel
import scipy
import torch

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")
# Move the model to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

voice_preset = "en_speaker_4"

inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

## move alll inputs to CUDA
for key in inputs:
  inputs[key] = inputs[key].to(device)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("output.wav", rate=sample_rate, data=audio_array)