
import argparse
from io import BytesIO
import os
import numpy as np
import sounddevice
import pyaudio; print(pyaudio.PyAudio().get_device_count())
import speech_recognition as sr
#import whisper
#import stable_whisper as whisper
#from faster_whisper import WhisperModel
import stable_whisper as whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from tts_wrap import TTSWrap

class RealtimeWhisper:

    def __init__(self,):
        self.model = "medium.en"
        self.non_english = False
        self.energy_threshold = 1000
        self.record_timeout = 3
        self.phrase_timeout = 2
        self.phrase_time = None
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        self.audio_model = whisper.load_faster_whisper(self.model, device="cuda", compute_type="float16")
        self.transcription = ['']
        self.last_work = datetime.utcnow()

        self.last_data_recorded = datetime.utcnow()
        self.last_data_recorded_threshold = 2 #sec
        self.data_processed = datetime.utcnow()
        self.first_data_recorded = self.data_processed#datetime.utcnow()
        self.first_start = True
        self.is_processed_queue = False

        self.generated_audio_queue = Queue()
        self.tts_model = TTSWrap()
        
        generated_speech = self.tts_model.voice("Hello, ask me a question")
        data = BytesIO(generated_speech.tobytes())
        print(f"total size {data.getbuffer().nbytes}")
        self.generated_audio_queue.put(generated_speech)
    
    def record_callback(self, audio_data) -> None:
        # Grab the raw bytes and push it into the thread safe queue.
        #data = audio.get_raw_data()
        self.data_queue.put(audio_data)
        self.last_data_recorded = datetime.utcnow()
        if self.data_processed >= self.first_data_recorded:
            self.first_data_recorded = datetime.utcnow()
            print("person started talk")
            self.first_start = False
            self.is_processed_queue = False

        print("put data in queue")
        
    def transcribe_progress_callback(self, processed_second_stamp, total_seconds):
        print(f"processed {processed_second_stamp}, total {total_seconds}")


    def work_loop(self,) -> None:
        while True:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                last_record_delta = now - self.last_data_recorded
                if self.first_start:
                    sleep(0.25)
                    continue

                if last_record_delta <= timedelta(seconds=self.phrase_timeout):
                    print(f"waiting for the person to finish the conversation {last_record_delta.total_seconds()}, threshold {timedelta(seconds=self.phrase_timeout).total_seconds()}")
                    sleep(0.25)
                    continue
                
                if self.is_processed_queue:
                    sleep(0.25)
                    continue

                print("person finished talk")
                delta = now - self.last_work
                if delta > timedelta(seconds=self.phrase_timeout):
                    self.last_work = datetime.utcnow()
                    if not self.data_queue.empty():
                        phrase_complete = False
                        # If enough time has passed between recordings, consider the phrase complete.
                        # Clear the current working audio buffer to start over with the new data.
                        if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                            phrase_complete = True
                        # This is the last time we received new audio data from the queue.
                        self.phrase_time = now
                        
                        # Combine audio data from queue
                        audio_data = b''.join(self.data_queue.queue)
                        self.data_queue.queue.clear()
                        
                        # Convert in-ram buffer to something the model can use directly without needing a temp file.
                        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                        # Read the transcription.
                        print("transcribing")
                        result, info = self.audio_model.transcribe(audio_np)#, progress_callback=self.transcribe_progress_callback)

                        recognized_text = []

                        for i in result:
                            recognized_text.append(i.text)

                        recognized_text = ''.join(recognized_text)

                        print("recognized text", recognized_text)
                        
                        generated_speech = self.tts_model.voice(recognized_text)
                        data = BytesIO(generated_speech.tobytes())
                        print(f"total size {data.getbuffer().nbytes}")
                        self.generated_audio_queue.put(generated_speech)

                        self.data_processed = datetime.utcnow()
                        self.is_processed_queue = True
                        #text = list(result) #['text'].strip()
                        #print(f"transcribed {text}")

                        # If we detected a pause between recordings, add a new item to our transcription.
                        # Otherwise edit the existing one.
                        #if phrase_complete:
                        #    self.transcription.append(text)
                        #else:
                        #    self.transcription[-1] = text

                        # Clear the console to reprint the updated transcription.
                        #os.system('cls' if os.name=='nt' else 'clear')
                        #for line in self.transcription:
                        #    print(line)
                        # Flush stdout.
                        #print('', end='', flush=True)
                    else:
                        # Infinite loops are bad for processors, must sleep.
                        sleep(0.25)
                    
            except KeyboardInterrupt:
                ...
