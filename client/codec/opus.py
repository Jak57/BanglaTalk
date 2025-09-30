import opuslib

import logging
logging.basicConfig(level=logging.INFO)
# logging.disable(logging.INFO)

class OPUS_CODEC():
    def __init__(self, stream_info):
        self.sampling_rate = stream_info['sampling_rate']
        self.processing_time_ms = stream_info['processing_time_ms']
        self.channel = stream_info['channel']
        self.chunk_size = int((self.sampling_rate * self.channel * self.processing_time_ms) / 1000.0)

        self.encoder = opuslib.Encoder(
            self.sampling_rate,
            self.channel,
            opuslib.APPLICATION_VOIP
        )
        self.encoder.bitrate = 24000
        self.encoder.complexity = 10
        self.encoder.vbr = True
        self.encoder.signal = opuslib.SIGNAL_VOICE

        self.decoder = opuslib.Decoder(
            self.sampling_rate,
            self.channel
        )
        logging.info("--------------- Opus Codec Initialization Done -------------------")
    
    def encode(self, frame, chunk_size):
        """
        Encodes audio frame.

        Args:
            - frame (np.ndarray (int16)): Raw audio.
        Return:
            - byte array: Opus encoded audio.
        """
        return self.encoder.encode(frame.tobytes(), chunk_size)

    def decode(self, encoded_frame, chunk_size):
        """
        Decode Opus encoded audio data.

        Args:
            - encoded_frame (byte array): Opus encoded packet.
        Returns:
            - byte array: Decoded raw data.
        """
        return self.decoder.decode(encoded_frame, chunk_size, decode_fec=False)