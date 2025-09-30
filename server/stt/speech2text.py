## Local
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2ProcessorWithLM
import torch
import queue
import logging
logging.basicConfig(level=logging.INFO)
# logging.disable(logging.INFO)
import time
import numpy as np
import pyctcdecode
import librosa

# Speech to Text conversion system
class STT():
    def __init__(self, sampling_rate):
        """
        Loads STT model.

        Args:
            - sampling_rate (int): Sample rate of the audio stream.
        """
        self.sampling_rate = sampling_rate
        self.channel = 1
        self.sample_size_byte = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stt_queue = queue.Queue()
        self.stop_signal = object()
        self.silence_timestamp = None

        self.model_path_wav2vec2 = "model/stt/indicwav2vec_v1_bengali"
        self.model_path_bangal_regional_dialect_wav2vec2 = "model/stt/bangla_regional_dialect/wav2vec2_bangla_regional_dialect.pth"
        self.kenlm_model_path = "model/stt/kenlm/5gram_correct.arpa"

        self.processor = AutoProcessor.from_pretrained(self.model_path_wav2vec2)
        self.model = AutoModelForCTC.from_pretrained(self.model_path_wav2vec2)
        self.model.load_state_dict(torch.load(self.model_path_bangal_regional_dialect_wav2vec2)["model"])

        self.vocab_dict = self.processor.tokenizer.get_vocab()
        self.sorted_vocab_dict = {k: v for k, v in sorted(self.vocab_dict.items(), key=lambda item: item[1])}
        self.decoder = pyctcdecode.build_ctcdecoder(
            list(self.sorted_vocab_dict.keys()),
            str(self.kenlm_model_path)
        )
        self.processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            decoder=self.decoder
        )
        self.model.freeze_feature_encoder()

        self.model.to(self.device)
        self.model.eval()
        logging.info("---------------------- STT Model Loaded ------------------------")

    def inference(self, frame):
        """
        Performs inference for speech to text conversion.

        Args:
            - frame (numpy array): Audio frame
        Returns:
            - str: Returns transcription.
        """

        logging.info(f"device={self.device} model_device={self.model.device}")
        t1 = time.time()
        try:
            inputs = self.processor(
                frame,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=False
            )

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            preds = self.processor.batch_decode(predicted_ids)[0]
            text = preds.replace("[PAD]", "")
            logging.info(f"STT: time taken={(time.time() - t1) * 1000:.2f} ms and text={text}")
            return text
        except Exception as e:
            logging.info(f"STT: Error e={e}")
            return ""

    def inference_kenlm(self, frame):
        """
        Performs inference for speech to text conversion.

        Args:
            - frame (numpy array): Audio frame
        Returns:
            - str: Returns transcription.
        """

        logging.info(f"device={self.device} model_device={self.model.device}")
        t1 = time.time()
        try:
            inputs = self.processor(
                frame,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=False
            )

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits

            np_logits = logits.squeeze(0).cpu().numpy()
            result = self.processor_with_lm.decode(np_logits, beam_width=256)
            text = result.text
            
            logging.info(f"STT: time taken={(time.time() - t1) * 1000:.2f} ms and text={text}")
            return text
        except Exception as e:
            logging.info(f"STT: Error e={e}")
            return ""
    
    def process(self, audio_manager, vad):
        """
        Converts speech to text.

        Args:
            audio_manager: Instance of Audio Manager.
            vad: Instance of Silero VAD.
        """
        buffer = []
        speech = 0
        speech_segments = []
        isStartFound = False
        timer = None
        start_timestamp = None
        while True:
            try:
                arr_i16 = audio_manager.audio_queue.get_nowait()
                frame = (arr_i16.astype(np.float32) / 32768.0).tolist()

                if frame is audio_manager.stop_signal:
                    if len(speech_segments) > 0:
                        self.stt_queue.put(np.array(speech_segments))
                        self.stt_queue.put(self.stop_signal)
                    break
                buffer += frame
                if len(buffer) >= vad.WINDOW_SIZE_SAMPLES:
                    subframe = buffer[:vad.WINDOW_SIZE_SAMPLES]
                    buffer = buffer[vad.WINDOW_SIZE_SAMPLES:]
                    speech_dict = vad.get_speech_dict(torch.tensor(subframe))
                    if not speech_dict and timer:
                        if (time.time() - timer) >= vad.silence_threshold_ms/1000:
                            logging.info(f"-------------------------------> Silence found, duration={time.time() - timer:.2f} s time={time.time()}\n")
                            self.silence_timestamp = time.time()

                            self.stt_queue.put(np.array(speech_segments))
                            speech_segments = []
                            self.stt_queue.put(self.stop_signal)
                            timer = None
                    if speech_dict:
                        speech += 1
                        logging.info(f"{speech}: {speech_dict}")
                        if list(speech_dict.keys())[0] == "start":
                            timer = None
                            isStartFound = True
                            start_timestamp = time.time()
                        elif list(speech_dict.keys())[0] == "end":
                            timer = time.time()
                            isStartFound = False

                    speech_segments += subframe
            except queue.Empty:
                time.sleep(0.001)

    def sender(self, llm):
        """
        Perform inference and send text to TTS system.
        """
        segments = []
        while True:
            try:
                frame = self.stt_queue.get_nowait()
                if frame is self.stop_signal and len(segments) > 0:
                    text = " ".join(segments)
                    logging.info(f"Stop signal found: Time={time.time()} s time_diff={(time.time() - self.silence_timestamp) * 1000:.2f} ms text={text}")
                    llm.llm_queue.put(text)
                    segments = []
                    continue

                if not (frame is self.stop_signal):
                    # # Inference with out KenLM
                    text = self.inference(frame)

                    # Inference with KenLM
                    # text = self.inference_kenlm(frame)

                    if text and len(text) > 1:
                        segments.append(text)
            except queue.Empty:
                time.sleep(0.001)


if __name__ == "__main__":
    path = "data/input/sylhet_1.wav"
    sampling_rate = 16000
    stt = STT(sampling_rate)

    y, sr = librosa.load(path, sr=sampling_rate, mono=True)

    t1 = time.time()
    text = stt.inference_kenlm(y)
    # text = stt.inference(y)
    t2 = time.time()
    logging.info(f"Text={text} and time taken={(t2-t1):.2f} s")
