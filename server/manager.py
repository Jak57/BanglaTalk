import queue
import logging
logging.basicConfig(level=logging.INFO)
# logging.disable(logging.INFO)
import soundfile as sf
import time
import numpy as np
import random

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
        self.audio_queue = queue.Queue()
        self.buffer_size = self.chunk_size * 2
        self.stop_signal = object()

        self.player_queue = queue.Queue()
        self.client_addr = None
        self.sequence_number = 1
        self.ssrc = random.randint(1, 100000)

        self.timestamp = None
        self.rtp_header_size = 12
        self.wav_file = sf.SoundFile("data/output/received.raw", mode="w", samplerate=16000, channels=1, subtype="PCM_16")
        logging.info("--------------- Audio Manager Initialization Done -------------------")

    def receive(self, server, opus_codec):
        """
        Receives audio frames from the client.
        """
        count = 1
        try:
            while True:
                data, addr = server.recvfrom(self.buffer_size + self.rtp_header_size)
                self.client_addr = addr
                count += 1
                if not data:
                    break

                data = data[self.rtp_header_size:]
                data = opus_codec.decode(data)
                arr_int16 = np.frombuffer(data, dtype=np.int16)
                
                self.wav_file.write(arr_int16)
                self.audio_queue.put(arr_int16)

                if count % 100 == 0:
                    logging.info(f"Received packet i={count}")

        except Exception as e:
            logging.info(f"Received error: {e}")

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
        return bytes(header) + data

    def send_audio_to_client(self, sock, opus_codec, llm):
        """
        Send response of LLM to the client.
        """
        while not self.client_addr:
            time.sleep(self.processing_time_ms / (2 * 1000))

        while True:
            if llm.is_interruption_found:
                while True:
                    try:
                        self.player_queue.get_nowait()
                    except queue.Empty:
                        break

            try:
                chunk = self.player_queue.get_nowait()
                chunk = np.array(chunk, dtype=np.float32).reshape(-1, 1)

                chunk = np.clip(chunk, -1.0, 1.0)         
                chunk = (chunk * 32767).astype(np.int16)
                encoded_data_byte = opus_codec.encoder.encode(chunk.tobytes(), self.chunk_size)

                try:
                    rtp_packet = self.prepare_rtp_packet(encoded_data_byte)
                    sock.sendto(rtp_packet, self.client_addr)
                    time.sleep(self.processing_time_ms / (1 * 1000) - 0.003)
                except Exception as e:
                    logging.info(e)
            except queue.Empty:
                time.sleep(0.001)