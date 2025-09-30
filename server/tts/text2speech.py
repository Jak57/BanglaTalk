## VITS
## --------------------------------------------------------------------- Local ---------------------------------------------------------------------
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import queue
import time
# from transformers import VitsModel, AutoTokenizer
from TTS.utils.synthesizer import Synthesizer
import torch

import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class TTS():
    def __init__(self):
        """
        Loads Phoneme and TTS models.
        """
        self.sampling_rate = 16000
        self.processing_time_ms = 20
        self.channel = 1
        self.chunk_size = int((self.sampling_rate * self.processing_time_ms * self.channel) / 1000)
        self.text_queue = queue.Queue()

        self.phoneme_queue = queue.Queue()
        self.timestamps = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vits_tts_sampling_rate = 22050
        self.vits_model_path = "model/tts/vits/male/checkpoint_910000.pth"
        self.vits_config_path = "model/tts/vits/male/config.json"
        self.use_cuda = True if torch.cuda.is_available() else False

        self.vits_model = Synthesizer(
            self.vits_model_path,
            self.vits_config_path,
            use_cuda=self.use_cuda
        )

        logging.info("--------------------- TTS Model Loaded -------------------------")

    # def inference_mms_tts_ben(self, text):
    #     try:
    #         inputs = self.tokenizer(text, return_tensors="pt")
    #         inputs = {k: v.to(self.device) for k, v in inputs.items()}

    #         with torch.inference_mode():
    #             out = self.model(**inputs)

    #         wave = out.waveform
    #         if wave.dim() == 2:
    #             wave = wave[0]
            
    #         y = wave.float().cpu().numpy()
    #         float_array = np.clip(y, -1.0, 1.0)
    #         return float_array
    #     except Exception:
    #         return None

    def inference_vits(self, text):

        try:
            # logging.info(f"TTS: text={text}")
            print(f"TTS: text={text}")

            x = self.vits_model.tts(text)
            # print(f"Type of x={type(x)} len={len(x)} {x[:10]}")

            # y = torch.as_tensor(x)
            # print(f"Type of y={type(y)} len={len(y)} {y[:10]}")

            # z = y.detach().cpu().numpy()
            # print(f"Type of z={type(z)} len={len(z)} {z[:10]}")

            x = np.array(x)
            # print(f"Type of a={type(a)} len={len(a)} {a[:10]}")

            x = librosa.resample(x, orig_sr=self.vits_tts_sampling_rate, target_sr=self.sampling_rate)
            return x
        except Exception:
            return None


    def process(self, audio_manager, llm):
        """
        Generates phoneme from the text.
        """
        cnt = 0
        while True:
            if llm.is_interruption_found:
                while True:
                    try:
                        self.text_queue.get_nowait()
                    except queue.Empty:
                        break

            try:
                text = self.text_queue.get_nowait()

                if text is audio_manager.stop_signal:
                    break

                if text == "$":
                    continue

                t1 = time.time()

                # float_array = self.inference_mms_tts_ben(text)
                float_array = self.inference_vits(text)

                logging.info(f"TTS: Time taken in TTS={(time.time() - t1) * 1000:.2f} ms and text={text}")

                if type(float_array) == None:
                    continue

                # if not float_array:
                #     continue

                cnt += 1
                total_sample = len(float_array)
                idx = 0
                isLeftOverFound = False
                while idx < total_sample:
                    if idx + self.chunk_size > total_sample:
                        last_chunk = float_array[idx:total_sample]
                        pad_length = self.chunk_size - len(last_chunk)
                        padded_chunk = np.pad(last_chunk, (0, pad_length), mode='constant', constant_values=0.0)
                        audio_manager.player_queue.put(padded_chunk)
                        break
                    frame = np.array(float_array[idx:idx + self.chunk_size])
                    audio_manager.player_queue.put(frame)
                    idx += self.chunk_size
            except queue.Empty:
                time.sleep(0.001)


## MMS
# ## --------------------------------------------------------------------- Local ---------------------------------------------------------------------
# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import queue
# import time
# from transformers import VitsModel, AutoTokenizer
# import torch

# import logging
# logging.basicConfig(level=logging.INFO)
# import numpy as np

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# class TTS():
#     def __init__(self):
#         """
#         Loads Phoneme and TTS models.
#         """
#         self.sampling_rate = 16000
#         self.processing_time_ms = 20
#         self.channel = 1
#         self.chunk_size = int((self.sampling_rate * self.processing_time_ms * self.channel) / 1000)
#         self.text_queue = queue.Queue()

#         self.phoneme_queue = queue.Queue()
#         self.timestamps = []

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_path = "model/tts/mms_tts_ben"
#         self.model = VitsModel.from_pretrained(
#             self.model_path,
#             dtype=torch.float32
#         ).to(self.device).eval()

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

#         logging.info("--------------------- TTS Model Loaded -------------------------")

#     def inference_mms_tts_ben(self, text):
#         try:
#             inputs = self.tokenizer(text, return_tensors="pt")
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.inference_mode():
#                 out = self.model(**inputs)

#             wave = out.waveform
#             if wave.dim() == 2:
#                 wave = wave[0]
            
#             y = wave.float().cpu().numpy()
#             float_array = np.clip(y, -1.0, 1.0)
#             return float_array
#         except Exception:
#             return None

#     def process(self, audio_manager, llm):
#         """
#         Generates phoneme from the text.
#         """
#         cnt = 0
#         while True:
#             if llm.is_interruption_found:
#                 while True:
#                     try:
#                         self.text_queue.get_nowait()
#                     except queue.Empty:
#                         break

#             try:
#                 text = self.text_queue.get_nowait()

#                 if text is audio_manager.stop_signal:
#                     break

#                 if text == "$":
#                     continue

#                 t1 = time.time()

#                 float_array = self.inference_mms_tts_ben(text)

#                 logging.info(f"TTS: Time taken in TTS={(time.time() - t1) * 1000:.2f} ms and text={text}")

#                 if type(float_array) == None:
#                     continue

#                 cnt += 1
#                 total_sample = len(float_array)
#                 idx = 0
#                 isLeftOverFound = False
#                 while idx < total_sample:
#                     if idx + self.chunk_size > total_sample:
#                         last_chunk = float_array[idx:total_sample]
#                         pad_length = self.chunk_size - len(last_chunk)
#                         padded_chunk = np.pad(last_chunk, (0, pad_length), mode='constant', constant_values=0.0)
#                         audio_manager.player_queue.put(padded_chunk)
#                         break
#                     frame = np.array(float_array[idx:idx + self.chunk_size])
#                     audio_manager.player_queue.put(frame)
#                     idx += self.chunk_size
#             except queue.Empty:
#                 time.sleep(0.001)