import soundfile as sf
import sounddevice as sd
import time
import queue
import random
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

class AudioManager():
    def __init__(self, stream_info):
        """
        Initilized audio processor.

        Args:
            - stream_info (dict): Contains information about the audio stream.
        """
        self.sampling_rate = stream_info['sampling_rate']
        self.processing_time_ms = stream_info['processing_time_ms']
        self.channel = stream_info['channel']
        self.chunk_size = int((self.sampling_rate * self.channel * self.processing_time_ms) / 1000.0)

        self.raw_audio_queue = queue.Queue()
        self.player_queue = queue.Queue()
        self.receiver_queue = queue.Queue()

        self.player_sample_rate = 16000
        self.player_chunk_size = int((self.player_sample_rate * self.channel * self.processing_time_ms) / 1000.0)
        self.player_buffer_size = self.player_chunk_size * 2

        self.rtp_header_size = 12
        self.sequence_number = 1
        self.ssrc = random.randint(1, 100000)
        logging.info("--------------- Audio Manager Initialization Done -------------------")
    
    def parse_rtp_header(self, header):
        """
        Parses RTP packet.

        Args:
            - data (byte array): Raw audio frame with header.
        """
        sequence_number = (header[2] << 8) | header[3]
        timestamp = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7]
        ssrc = (header[8] << 24) | (header[9] << 16) | (header[10] << 8) | header[11]
        logging.info(f"Received frame: sequence_number={sequence_number} timestamp={timestamp} ssrc={ssrc}")

    def prepare_rtp_packet(self, data):
        """
        Prepares RTP packet.
        
        Args:
            - data (byte array): Raw PCM data.
        Returns:
            - byte array: Rtp frame with header.
        """
        header_size = 12
        codec = 1
        timestamp = int(time.time() * 1000)
        header = bytearray(header_size)
        header[0] = 128
        header[1] = codec & 0xFF
        header[2] = (self.sequence_number >> 8) & 0xFF
        header[3] = self.sequence_number & 0xFF
        header[4] = (timestamp >> 24) & 0xFF
        header[5] = (timestamp >> 16) & 0xFF
        header[6] = (timestamp >> 8) & 0xFF
        header[7] = timestamp & 0xFF
        header[8] = (self.ssrc >> 24) & 0xFF
        header[9] = (self.ssrc >> 16) & 0xFF
        header[10] = (self.ssrc >> 8) & 0xFF
        header[11] = self.ssrc & 0xFF
        self.sequence_number += 1
        if self.sequence_number > 65535:
            self.sequence_number = 1
        return bytes(header) + data

    def process_frame(self, frame, drc, rnnoise, opus):
        """
        Process raw audio frame and encode it.

        Args:
            - frame (np.ndarray (int16)): Raw audio.
            - drc (class): DRC object.
            - rnnoise (class): RNNoise object.
            - opus (class): Opus object.
        """
        frame = drc.process_drc(frame)
        frame = rnnoise.process_rnnoise(frame)
        encoded_frame = opus.encode(frame, self.chunk_size)
        return encoded_frame

    def send_to_server(self, sock, server_addr, drc, rnnoise, opus):
        """
        Sends UDP packet to the server.

        Args:
            - sock: Datagram socket.
            - server_addr (tuple): Server IP and Port.
            - drc (class): DRC object.
            - rnnoise (class): RNNoise object.
            - opus (class): Opus object.
        """
        i = 0
        while True:
            try:
                frame = self.raw_audio_queue.get_nowait()
                encoded_frame = self.process_frame(frame, drc, rnnoise, opus)
                rtp_packet = self.prepare_rtp_packet(encoded_frame)
                try:
                    sock.sendto(rtp_packet, server_addr)
                    logging.info(f"Sent frame={i} len={len(rtp_packet)}")
                except Exception as e:
                    logging.info(e)
                i += 1
            except queue.Empty:
                time.sleep(0.001)

    def file_reader(self, path):
        """
        Read audio from file.

        Args:
            - path (str): Audio file path.
        """
        with sf.SoundFile(path, 'r') as f:
            logging.info("----------------- File Reader Thread Started -----------------------")
            while True:
                frame = f.read(self.chunk_size, dtype='int16')
                if len(frame) < self.chunk_size:
                    break
                self.raw_audio_queue.put(frame)
                time.sleep(self.processing_time_ms/1000)
    
    def recorder(self):
        """
        Records audio from microphone and send to the network.
        """
        def audio_callback_mic(indata, frames, time_info, status):
            self.raw_audio_queue.put(indata)
        logging.info("----------------- Recording started... -----------------------")
        with sd.InputStream(samplerate=self.sampling_rate, channels=self.channel, blocksize=self.chunk_size, dtype='int16', callback=audio_callback_mic):
            while True:
                time.sleep(0.001)
        
    def receiver(self, sock, opus):
        """
        Receives audio stream from the server.

        Args:
            - sock: Datagram socket.
        """
        while True:
            try:
                data = sock.recv(self.player_buffer_size + self.rtp_header_size)
                self.parse_rtp_header(data[:self.rtp_header_size])
                encoded_data_byte = data[self.rtp_header_size:]
                data = opus.decode(encoded_data_byte, self.chunk_size)
                samples = np.frombuffer(data, dtype='int16')
                samples = (samples / 32768.0).astype(np.float32)
                self.receiver_queue.put(samples.reshape(-1, 1))
            except:
                time.sleep(0.001)

    def player(self):
        """
        Plays audio stream.
        """
        def audio_callback(outdata, frames, time_info, status):
            try:
                outdata[:] = self.receiver_queue.get_nowait()
            except Exception as e:
                outdata[:] = np.zeros((self.player_chunk_size, 1), dtype='float32')
                
        with sd.OutputStream(self.player_sample_rate, blocksize=self.player_chunk_size, channels=self.channel, dtype='float32', callback=audio_callback):
            while True:
                time.sleep(0.001)