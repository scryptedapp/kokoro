from __future__ import annotations
from typing import Any
import scrypted_sdk
import asyncio
import base64
import io
import platform
from scrypted_sdk.types import LLMTools, ChatCompletionFunctionTool, MediaConverter, MediaObjectOptions
from kokoro import KPipeline
import soundfile as sf

class KokoroPlugin(scrypted_sdk.ScryptedDeviceBase, LLMTools, MediaConverter):
    def __init__(self):
        super().__init__()
        self.pipeline: KPipeline | None = None
        self.converters = [
            ["text/plain", "audio/ogg"],
            ["text/plain", "audio/wav"],
            ["text/plain", "audio/mpeg"],
            ["text/plain", "audio/flac"],
            ["text/plain", "audio/mp3"],
            ["text/plain", "audio/mpeg"],
        ]

    async def convertMedia(self, data: Any, fromMimeType: str, toMimeType: str, options: MediaObjectOptions = None) -> Any:
        text: str = data
        
        # Map MIME types to soundfile formats
        mime_to_format = {
            "audio/wav": "WAV",
            "audio/x-wav": "WAV", 
            "audio/wave": "WAV",
            "audio/ogg": "OGG",
            "audio/mpeg": "MP3",
            "audio/mp3": "MP3",
            "audio/flac": "FLAC",
            "audio/x-flac": "FLAC"
        }
        
        # Check if the requested format is supported
        if toMimeType not in mime_to_format:
            supported_formats = ", ".join(mime_to_format.keys())
            raise ValueError(f"Unsupported audio format: {toMimeType}. Supported formats: {supported_formats}")
        
        audio_format = mime_to_format[toMimeType]
        
        # Initialize pipeline if needed
        if not self.pipeline:
            device = "mps" if platform.system() == "Darwin" and platform.machine() == "arm64" else None
            self.pipeline = KPipeline(lang_code='a', device=device)
        
        # Generate audio without splitting text
        generator = self.pipeline(text, voice='af_heart', split_pattern=None)
        
        # Process the single audio result
        full_audio = None
        for i, (gs, ps, audio) in enumerate(generator):
            full_audio = audio
            break  # Only one result expected
        
        if full_audio is None:
            raise ValueError("No audio was generated")
        
        # Create in-memory buffer and write audio
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format=audio_format)
        buffer.seek(0)
        
        # Return the audio bytes
        return buffer.getvalue()

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
            device = "mps" if platform.system() == "Darwin" and platform.machine() == "arm64" else None
            self.pipeline = KPipeline(lang_code='a', device=device)

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
