from pyrnnoise import RNNoise
import numpy as np

class RNNOISE_CANCELLER():

    def __init__(self, stream_info):
        """
        Initialized RNNoise object.

        Args:
            - stream_info (dict): Contains information about the audio stream.
        """
        self.stream_sampling_rate = stream_info['sampling_rate']
        self.channel = stream_info['channel']
        self.stream_processing_time = stream_info['processing_time_ms']
        self.stream_chunk_size = int((self.stream_sampling_rate * self.stream_processing_time * self.channel) / 1000)

        self.rnnoise_sampling_rate = 48000
        self.rnnoise_processing_time_ms = 10
        self.rnnoise = RNNoise(sample_rate=self.rnnoise_sampling_rate)
        self.rnnoise_chunk_size = int((self.rnnoise_sampling_rate * self.rnnoise_processing_time_ms * self.channel) / 1000)
        self.rnnoise_ratio = int(self.rnnoise_sampling_rate / self.stream_sampling_rate)
        self.rnnoise_prev_value = 0

    def upsample(self, frame):
        """
        Performs upsampling.

        Args:
            - frame (arry of int16): Input array.
        Returns:
            - np.ndarray: Upsample array
        """
        N = self.rnnoise_ratio * len(frame)
        arr = np.zeros(N, dtype='int16')
        idx = 0
        for i in range(len(frame)):
            cur = frame[i]
            delta = (cur - self.rnnoise_prev_value) / self.rnnoise_ratio
            
            for j in range(self.rnnoise_ratio):
                upsampled_value = int(self.rnnoise_prev_value + (j * delta))
                if upsampled_value > 32767:
                    upsampled_value = 32767

                if upsampled_value < -32768:
                    upsampled_value = -32768

                arr[idx] = upsampled_value
                idx += 1
            self.rnnoise_prev_value = cur
        return arr

    def downsample(self, frame):
        """
        Performs downsampling.

        Args:
            - frame (np.ndarray): Upsampled audio.
        Returns:
            - np.ndarray: Downsampled audio.
        """
        N = int(len(frame) / self.rnnoise_ratio)
        arr = np.zeros(N, dtype='int16')
        idx = 0
        start_idx = 0
        while True:
            if start_idx + self.rnnoise_ratio > len(frame):
                break
            arr[idx] = frame[start_idx]
            idx += 1
            start_idx += self.rnnoise_ratio
        return arr
    
    def process_rnnoise(self, frame):
        """
        Performs noise cancellation.

        Args:
            - frame (np.ndarray (int16)): Raw audio.
        Returns:
            - np.ndarray (int16): Noise cancelled audio. 
        """
        subframe = self.upsample(frame)
        idx = 0
        upsampled_list = []
        while True:
            if idx + self.rnnoise_chunk_size > len(subframe):
                break
            frame_rnn = subframe[idx:idx + self.rnnoise_chunk_size]
            for prob, arr_denoised in self.rnnoise.denoise_chunk(frame_rnn):
                arr_denoised = np.squeeze(arr_denoised, axis=0)
                for val in arr_denoised:
                    upsampled_list.append(val)
            idx = idx + self.rnnoise_chunk_size
        denoised_processed_upsampled_audio = np.array(upsampled_list)
        return self.downsample(denoised_processed_upsampled_audio)
