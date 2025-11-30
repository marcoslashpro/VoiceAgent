from typing import AsyncGenerator
from pipecat.frames.frames import (
    TTSStartedFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    ErrorFrame,
    Frame,
)
from pipecat.services import ollama
from pipecat.adapters.services.open_ai_adapter import (
    OpenAILLMInvocationParams
)
from pipecat.services.whisper import WhisperSTTService
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_filter import BaseTextFilter
from kokoro import KPipeline
import numpy as np
import torch

from pydantic import BaseModel


KOKORO_SAMPLE_RATE = 24000  # KOKORO's required sample rate

MONO = 1

# The Whisper model requires the audio data to be represented as 32-bit floating-point numbers (float32),
# normalized between -1.0 and 1.0.
# Pipecat's AudioFrames store samples as 16-bit integers.
# The normalization factor for 16-bit PCM is 2^15 (or 32768).
# *According to Gemini
INT_NORMALIZATION_FACTOR = 32768.0


class OllamaLLMService(ollama.OLLamaLLMService):
    def __init__(
        self,
        output_format: type[BaseModel] | None = None,
        think_enabled: bool = False,
        *,
        model: str = "llama2",
        base_url: str = "http://localhost:11434/v1",
        **kwargs,
    ):
        super().__init__(model=model, base_url=base_url, **kwargs)
        self._think_enabled = think_enabled
        self._output_format = output_format

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        if self._output_format is not None:
            params["format"] = self._output_format.model_json_schema()
        return params


class KokoroTTSService(TTSService):
    def __init__(self, pipeline: KPipeline, voice: str, text_filters: list[BaseTextFilter]) -> None:
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
