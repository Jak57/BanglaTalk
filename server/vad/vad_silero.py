from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    VADIterator
)
import logging
logging.basicConfig(level=logging.INFO)
# logging.disable(logging.INFO)

## Silero VAD (Voice Activity Detection)
class SileroVAD():
    def __init__(self, sampling_rate, silence_threshold_ms):
        """
        Loads Silero VAD.

        Args:
            - sampling_rate (int): Sample rate of the audio stream.
            - silence_threshold_ms (int): todo
        """
        self.WINDOW_SIZE_SAMPLES = 512
        self.sampling_rate = sampling_rate
        self.silence_threshold_ms = silence_threshold_ms
        self.model = load_silero_vad(onnx=True)
        self.iterator = VADIterator(self.model, sampling_rate=sampling_rate)
        logging.info("------------------------- Silero VAD Loaded -------------------------")

    def get_speech_dict(self, frame):
        """
        Provide information about the frame whether it is the start of the speech, end, silence or middle of the speech.

        Args:
            - frame (torch): Audio frame of 512 samples.
        Returns:
            - dict: Return a dictionary of key value pairs. Where key is start/end and value is timestamp in second. Dictionary can be None, in the middle of the speech or when there is silence.
        """
        return self.iterator(frame, return_seconds=True)