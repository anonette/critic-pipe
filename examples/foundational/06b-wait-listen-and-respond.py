#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    print(f"!!! MetricsFrame: {frame}, ttfb: {d.value}")
                elif isinstance(d, ProcessingMetricsData):
                    print(f"!!! MetricsFrame: {frame}, processing: {d.value}")
                elif isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    print(
                        f"!!! MetricsFrame: {frame}, tokens: {tokens.prompt_tokens}, characters: {tokens.completion_tokens}"
                    )
                elif isinstance(d, TTSUsageMetricsData):
                    print(f"!!! MetricsFrame: {frame}, characters: {d.value}")
        await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        ml = MetricsLogger()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                ml,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # We'll wait for the participant to talk before starting the conversation
            logger.info("Waiting for participant to speak before starting the conversation...")

        @transport.event_handler("on_transcription_message")
        async def on_transcription_message(transport, message):
            # Check if this is a final transcription message
            if message.get("is_final", False):
                text = message.get("text", "").strip()
                if text:
                    logger.info(f"Participant said: {text}")
                    # Add the user's message to the conversation
                    messages.append({"role": "user", "content": text})
                    # Add a system message to instruct the bot to respond
                    messages.append({"role": "system", "content": "Please respond to what the user just said."})
                    # Queue the context frame to trigger the bot's response
                    await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
