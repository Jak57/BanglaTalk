import threading
import socket

import torch
import gc
import time

from config import KEY
from manager import AudioManager
from codec.opus import OPUS_AUDIO_CODEC
from vad.vad_silero import SileroVAD

from stt.speech2text import STT
from llm.llm_response import LLM
from tts.text2speech import TTS

import logging
logging.basicConfig(level=logging.INFO)

gc.collect()
torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.info("------------------ BanglaTalk Server started... ---------------------")
    stream_info = {
        'sampling_rate': 16000,
        'processing_time_ms': 20,
        'channel': 1,
    }

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**10)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2**10)

    server_ip = KEY['SERVER_IP']
    server_port = KEY['SERVER_PORT']
    server.bind((server_ip, server_port))
    logging.info(f"Server listening on port={server_port}...\n\n")

    silence_threshold_ms = 1200
    manager = AudioManager(stream_info)
    opus_codec = OPUS_AUDIO_CODEC(stream_info)
    vad = SileroVAD(stream_info['sampling_rate'], silence_threshold_ms)
    stt = STT(stream_info['sampling_rate'])
    llm = LLM()
    tts = TTS()

    audio_receiver_thread = threading.Thread(target=manager.receive, args=(server, opus_codec))
    text2speech_process_thread = threading.Thread(target=stt.process, args=(manager, vad))
    text2speech_sender_thread = threading.Thread(target=stt.sender, args=(llm,))
    llm_process_thread = threading.Thread(target=llm.get_response, args=(tts,))
    tts_process_thread = threading.Thread(target=tts.process, args=(manager, llm))
    audio_sender_thread = threading.Thread(target=manager.send_audio_to_client, args=(server, opus_codec, llm))

    audio_receiver_thread.start()
    text2speech_process_thread.start()
    text2speech_sender_thread.start()
    llm_process_thread.start()
    tts_process_thread.start()
    audio_sender_thread.start()

    audio_receiver_thread.join()
    text2speech_process_thread.join()
    text2speech_sender_thread.join()
    llm_process_thread.join()
    tts_process_thread.join()
    audio_sender_thread.join()