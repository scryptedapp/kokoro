from __future__ import annotations
import scrypted_sdk
import asyncio
import base64
import io
from scrypted_sdk.types import LLMTools, ChatCompletionFunctionTool
from kokoro import KPipeline
import soundfile as sf

class KokoroPlugin(scrypted_sdk.ScryptedDeviceBase, LLMTools):
    def __init__(self):
        super().__init__()
        self.pipeline: KPipeline | None = None

    async def getLLMTools(self) -> list[ChatCompletionFunctionTool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate-audio",
                    "description": "Generate audio based on a text prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text  to generate the audio from.",
                            },
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def callLLMTool(self, name, parameters):
        if name == "generate-audio":
            return await self._handle_generate_audio(parameters)
        else:
            raise ValueError(f"Unknown tool name: {name}")

    async def _handle_generate_audio(self, parameters):
        # Extract parameters with defaults and validate first
        text = parameters.get("text")
        if not text:
            raise ValueError("Text is required for audio generation")


        if not self.pipeline:
            self.pipeline = KPipeline(lang_code='a')

        # Start timer for generation
        generator = self.pipeline(text, voice='af_heart')
        audio_segments = []
        
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            
            # Create in-memory buffer for audio data
            buffer = io.BytesIO()
            sf.write(buffer, audio, 24000, format='OGG')
            buffer.seek(0)
            
            # Convert to base64
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            audio_segments.append({
                "type": "audio", 
                "data": base64_data, 
                "mimeType": "audio/ogg"
            })

        # Return the tool result with array of audio segments
        return {
            "content": audio_segments
        }

def create_scrypted_plugin():
    return KokoroPlugin()
