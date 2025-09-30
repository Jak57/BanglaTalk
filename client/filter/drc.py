import math

class DYNAMIC_RANGE_COMPRESSION():
    def __init__(self):
        """
        Initialized DRC.
        """
        self.threshold = -10
        self.ratio = 2
        self.EPS = 1e-8

    def process_drc(self, frame):
        """
        Compresses the dynamic range of the audio segment.

        Args:
            - frame (np.ndarray (int16)): Audio segment
        Returns:
            - np.ndarray (int16): Range compressed audio. 
        """
        for i in range(len(frame)):
            sample_value = frame[i]
            if sample_value == 0:
                continue

            sample_normalized = math.fabs(sample_value / 32768.0)
            decibel = 20 * math.log10(max(sample_normalized, self.EPS))

            if decibel <= self.threshold:
                continue

            decibel_new = self.threshold + (decibel - self.threshold) / self.ratio
            is_sign_positive = 1.0
            if sample_value < 0:
                is_sign_positive = -1.0

            frame[i] = int(math.pow(10, (decibel_new / 20)) * is_sign_positive * 32767.0)
        return frame