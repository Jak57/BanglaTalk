import socket
import threading

from manager import AudioManager
from filter.drc import DYNAMIC_RANGE_COMPRESSION
from rnnoise.noise_canceller import RNNOISE_CANCELLER
from codec.opus import OPUS_CODEC   
from config import KEY

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("------------------ BanglaTalk Client started... ---------------------\n")

    stream_info = {
        # 'test_path': "data/input/mymensingh_1.wav",
        'test_path': "data/input/sylhet_1.wav",
        # 'test_path': "data/input/chittagong_1.wav",
        'sampling_rate': 16000,
        'processing_time_ms': 20,
        'channel': 1,
    }

    manager = AudioManager(stream_info)
    drc = DYNAMIC_RANGE_COMPRESSION()
    rnnoise = RNNOISE_CANCELLER(stream_info)
    opus = OPUS_CODEC(stream_info)

    server_addr = (KEY["SERVER_IP"], KEY["SERVER_PORT"])
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
    recv_sock.bind(("0.0.0.0", KEY["CLIENT_RECV_PROT"]))
    logging.info("Connected to the server...\n")

    is_read_from_file = False
    if is_read_from_file:
        file_reader_thread = threading.Thread(target=manager.file_reader, args=(stream_info['test_path'],))
        sent_to_server_thread = threading.Thread(target=manager.send_to_server, args=(recv_sock, server_addr, drc, rnnoise, opus), daemon=True)
        receiver_thread = threading.Thread(target=manager.receiver, args=(recv_sock, opus), daemon=True)
        player_thread = threading.Thread(target=manager.player, daemon=True)

        file_reader_thread.start()
        sent_to_server_thread.start()
        receiver_thread.start()
        player_thread.start()

        file_reader_thread.join()
        sent_to_server_thread.join()
        receiver_thread.join()
        player_thread.join()
    else:
        recorder_thread = threading.Thread(target=manager.recorder)
        sent_to_server_thread = threading.Thread(target=manager.send_to_server, args=(recv_sock, server_addr, drc, rnnoise, opus), daemon=True)
        receiver_thread = threading.Thread(target=manager.receiver, args=(recv_sock, opus), daemon=True)
        player_thread = threading.Thread(target=manager.player, daemon=True)

        recorder_thread.start()
        sent_to_server_thread.start()
        receiver_thread.start()
        player_thread.start()

        recorder_thread.join()
        sent_to_server_thread.join()
        receiver_thread.join()
        player_thread.join()