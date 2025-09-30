import opuslib
import logging
logging.basicConfig(level=logging.INFO)

class OPUS_AUDIO_CODEC():

    def __init__(self, stream_info):
        """
        Initilize the Opus Codec.

        Args:
            - stream_info (dict): Contains information about the audio stream.
        """
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
    
    def encode(self, frame):
        """
        Encode audio frame.

        Args:
            - frame (np.ndarray (int16)): Raw audio data.
        Return:
            - byte array: Opus encoded data
        """
        return self.encoder.encode(frame.tobytes(), self.chunk_size)

    def decode(self, encoded_frame):
        """
        Decode Opus encoded audio data.

        Args:
            - encoded_frame (byte array): Opus encoded packet.
        Returns:
            - byte array: Decoded raw data.
        """
        return self.decoder.decode(encoded_frame, self.chunk_size, decode_fec=False)