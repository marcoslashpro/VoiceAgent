from typing import AsyncGenerator
from pipecat.frames.frames import (
    TTSStartedFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    ErrorFrame,
    Frame,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_filter import BaseTextFilter
from kokoro import KPipeline
import numpy as np
import torch


KOKORO_SAMPLE_RATE = 24000  # KOKORO's required sample rate

MONO = 1

# The Whisper model requires the audio data to be represented as 32-bit floating-point numbers (float32),
# normalized between -1.0 and 1.0.
# Pipecat's AudioFrames store samples as 16-bit integers.
# The normalization factor for 16-bit PCM is 2^15 (or 32768).
# *According to Gemini
INT_NORMALIZATION_FACTOR = 32768.0


class KokoroTTSService(TTSService):
    def __init__(
        self, pipeline: KPipeline, voice: str, text_filters: list[BaseTextFilter]
    ) -> None:
        super().__init__(text_filters=text_filters)
        self._pipeline = pipeline
        self._voice = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        generator = self._pipeline(text, voice=self._voice)
        yield TTSStartedFrame()

        try:
            for _, _, audio_tensor in generator:
                assert isinstance(
                    audio_tensor, torch.FloatTensor
                ), f"Wrong audio type, expented a torch FloatTensor but got: {type(audio_tensor)}"

                normalized_audio: np.ndarray = (
                    audio_tensor.numpy() * INT_NORMALIZATION_FACTOR
                ).astype(np.int16)

                yield TTSAudioRawFrame(
                    audio=normalized_audio.tobytes(),
                    sample_rate=KOKORO_SAMPLE_RATE,
                    num_channels=MONO,
                )
        except Exception as e:
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()
