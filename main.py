print("Loading Local Smart Turn Analyzer V3...")
from typing import cast
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

print("✅ Local Smart Turn Analyzer V3 loaded")
print("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

print("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from voiceagent.agents import WhisperSTTService, OLLamaLLMService, KokoroTTSService
from kokoro import KPipeline


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    print(f"Starting bot")

    stt = WhisperSTTService(model="tiny")

    tts = KokoroTTSService(pipeline=KPipeline(lang_code="a"), voice="af_heart")

    llm = OLLamaLLMService(model="smollm2:1.7b")

    messages = [
        {
            "role": "system",
            "content": "You are helpful. honest and harmless.",
        },
    ]

    context = LLMContext(cast(list[LLMContextMessage], messages))
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        print(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Say hello and briefly introduce yourself."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
