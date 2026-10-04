#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["websockets>=13", "numpy", "soundfile", "httpx"]
# ///
"""Stream an audio file to the deployed Kyutai STT WebSocket in real time and print the text.

./stream_file.py https://<web_url> [--audio-url ...] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time

import httpx
import numpy as np
import soundfile as sf
import websockets

SAMPLE_RATE = 24_000
FRAME = 1920
DEFAULT_AUDIO = "https://github.com/kyutai-labs/delayed-streams-modeling/raw/refs/heads/main/audio/bria.mp3"


def load_pcm(url: str) -> bytes:
    data = httpx.get(url, follow_redirects=True).content
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        idx = np.arange(0, len(audio), sr / SAMPLE_RATE)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def run(base_url: str, pcm: bytes, speed: float) -> None:
    ws_url = base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    t0 = time.perf_counter()
    async with websockets.connect(ws_url, max_size=None) as ws:
        first = json.loads(await ws.recv())
        if first["type"] != "ready":
            sys.exit(f"server said {first}")

        async def send() -> None:
            step = FRAME * 2
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i : i + step])
                await asyncio.sleep(FRAME / SAMPLE_RATE / speed)
            await ws.send(json.dumps({"type": "end"}))

        sender = asyncio.create_task(send())
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "word":
                print(msg["text"], end="", flush=True)
            elif msg["type"] == "end_of_turn":
                print(" ⏎", flush=True)
            elif msg["type"] == "done":
                break
        await sender
    print(f"\n[{len(pcm) / 2 / SAMPLE_RATE:.1f}s of audio in {time.perf_counter() - t0:.1f}s]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO)
    parser.add_argument("--speed", type=float, default=1.0, help="multiple of real time")
    args = parser.parse_args()
    asyncio.run(run(args.url, load_pcm(args.audio_url), args.speed))


if __name__ == "__main__":
    main()
