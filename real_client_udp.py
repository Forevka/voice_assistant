#!/usr/bin/python3

from io import BytesIO
import socket
import threading
import pyaudio
import platform
import signal
import math
import struct
import time
import os
import numpy as np

from protocol import DataType, Protocol


class Client:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connected = False
        self.name = "User" #input('Enter the name of the client --> ')

        while 1:
            try:
                self.target_ip = "127.0.0.1" #input('Enter IP address of server --> ')
                self.target_port = 9001 #int(input('Enter target port of server --> '))
                self.room = 1 #int(input('Enter the id of room  --> '))
                self.server = (self.target_ip, self.target_port)
                self.connect_to_server()
                break
            except Exception as err:
                print(err)
                print("Couldn't connect to server...")

        self.chunk_size = 512
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.threshold = 10
        self.short_normalize = (1.0 / 32768.0)
        self.swidth = 2
        self.timeout_length = 2

        # initialise microphone recording
        self.p = pyaudio.PyAudio()
        a = self.p.get_default_input_device_info()
        self.playing_stream = self.p.open(format=pyaudio.paFloat32, channels=self.channels, rate=self.rate, output=True, frames_per_buffer=self.chunk_size)
        self.recording_stream = self.p.open(format=self.audio_format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)

        # Termination handler
        def handler(signum, frame):
            print("\033[2KTerminating...")
            message = Protocol(dataType=DataType.Terminate, room=self.room, data=self.name.encode(encoding='UTF-8'))
            self.s.sendto(message.out(), self.server)
            if platform.system() == "Windows":
                os.kill(os.getpid(), signal.SIGBREAK)
            else:
                os.kill(os.getpid(), signal.SIGKILL)

        if platform.system() == "Windows":
            signal.signal(signal.SIGBREAK, handler)
        else:
            signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

        self.packets = 0
        
        # start threads
        self.s.settimeout(2)
        receive_thread = threading.Thread(target=self.receive_server_data).start()
        self.send_data_to_server()


    def receive_server_data(self):
        self.total_bytes = 0
        while self.connected:
            try:
                data, addr = self.s.recvfrom(514)
                message = Protocol(datapacket=data)
                print(f"received {len(data)}")
                if message.DataType == DataType.ClientData:
                    part = message.data#np.frombuffer(message.data, dtype=np.float32).tobytes()
                    self.total_bytes += BytesIO(part).getbuffer().nbytes
                    self.playing_stream.write(part)
                    print("User with id %s is talking (room %s), total bytes (%s), packets %s" % (message.head, message.room, self.total_bytes, self.packets))
                    self.packets += 1
                elif message.DataType == DataType.Handshake or message.DataType == DataType.Terminate:
                    print(message.data.decode("utf-8"))
            except socket.timeout:
                print("\033[2K", end="\r")  # clearing line
            except Exception as err:
                pass

    def connect_to_server(self):
        if self.connected:
            return True

        message = Protocol(dataType=DataType.Handshake, room=self.room, data=self.name.encode(encoding='UTF-8'))
        self.s.sendto(message.out(), self.server)

        data, addr = self.s.recvfrom(1026)
        datapack = Protocol(datapacket=data)

        if addr == self.server and datapack.DataType == DataType.Handshake:
            print('Connected to server to room %s successfully!' % datapack.room)
            print(datapack.data.decode("utf-8"))
            self.connected = True
        return self.connected

    def rms(self, frame):
        count = len(frame) / self.swidth
        format = "%dh" % count
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * self.short_normalize
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def record(self):
        current = time.time()
        end = time.time() + self.timeout_length

        while current <= end:
            data = self.recording_stream.read(self.chunk_size)
            if self.rms(data) >= self.threshold:
                end = time.time() + self.timeout_length
                try:
                    message = Protocol(dataType=DataType.ClientData, room=self.room, data=data)
                    self.s.sendto(message.out(), self.server)
                except:
                    pass
            current = time.time()

    def listen(self):
        while True:
            try:
                inp = self.recording_stream.read(self.chunk_size)
                rms_val = self.rms(inp)
                if rms_val > self.threshold:
                    self.record()
            except:
                pass

    def send_data_to_server(self):
        while self.connected:
            self.listen()


client = Client()
