"""Streaming speech-to-text with Kyutai STT, served over a WebSocket from one GPU.

Runs inside the container image that deploy.py builds for a Chalk ScalingGroup; the page
at `/` is index.html.

Wire protocol on `/ws`:
  client -> server  binary frames: 24 kHz mono little-endian int16 PCM, any chunk size
  client -> server  text {"type": "end"}: no more audio; flush the model's delay and finish
  server -> client  {"type": "ready"} once this connection owns the model
  server -> client  {"type": "busy"} then close, when another stream holds the model
  server -> client  {"type": "word", "text"}: a text piece, already space-prefixed per word
  server -> client  {"type": "end_of_turn"}: the model's semantic VAD saw the speaker stop
  server -> client  {"type": "done"} after an "end" has been fully flushed
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from moshi.models import LMGen, loaders

MODEL_NAME = os.environ.get("KYUTAI_MODEL", "kyutai/stt-1b-en_fr")
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")
# Probability above which the "end of turn" VAD head counts as the speaker having stopped.
VAD_THRESHOLD = 0.5

log = logging.getLogger("kyutai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class KyutaiSTT:
    """The model plus its streaming state. The state belongs to one audio stream at a time."""

    def __init__(self) -> None:
        super().__init__()
        t0 = time.monotonic()
        # A cancelled asyncio.to_thread keeps running, so a disconnect's reset() must wait
        # for any in-flight step rather than racing it.
        self.mu = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info = loaders.CheckpointInfo.from_hf_repo(MODEL_NAME)
        self.mimi = info.get_mimi(device=self.device)
        self.moshi = info.get_moshi(device=self.device)
        self.lm_gen = LMGen(self.moshi, temp=0, temp_text=0)
        self.tokenizer = info.get_text_tokenizer()
        self.sample_rate = int(self.mimi.sample_rate)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.silence_prefix_s = float(info.stt_config.get("audio_silence_prefix_seconds", 1.0))
        self.delay_s = float(info.stt_config.get("audio_delay_seconds", 5.0))
        self.padding_token_id = int(info.raw_config.get("text_padding_token_id", 3))

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        for _ in range(4):
            self.step_frame(np.zeros(self.frame_size, dtype=np.float32))
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.reset()
        log.info(
            "loaded %s on %s in %.1fs (frame=%d @ %d Hz, delay=%.1fs)",
            MODEL_NAME,
            self.device,
            time.monotonic() - t0,
            self.frame_size,
            self.sample_rate,
            self.delay_s,
        )

    def reset(self) -> None:
        with self.mu:
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()

    @torch.no_grad()
    def step_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Feeds exactly one frame (80 ms) of float32 PCM and returns the messages it produced."""
        chunk = torch.from_numpy(frame).to(self.device)[None, None, :]
        codes = self.mimi.encode(chunk)
        out: list[dict[str, Any]] = []
        for c in range(codes.shape[-1]):
            res = self.lm_gen.step_with_extra_heads(codes[:, :, c : c + 1])
            if res is None:
                continue
            text_tokens, vad_heads = res
            token = int(text_tokens[0, 0, 0].item())
            # 0 is end-of-padding and padding_token_id is padding; neither is text.
            if token not in (0, self.padding_token_id):
                out.append({"type": "word", "text": self.tokenizer.id_to_piece(token).replace("▁", " ")})
            if vad_heads and vad_heads[2][0, 0, 0].item() > VAD_THRESHOLD:
                out.append({"type": "end_of_turn"})
        return out

    def step_frames(self, pcm: np.ndarray) -> list[dict[str, Any]]:
        """Feeds a whole number of frames; the caller keeps any remainder for the next call."""
        out: list[dict[str, Any]] = []
        with self.mu:
            for i in range(0, len(pcm), self.frame_size):
                out.extend(self.step_frame(pcm[i : i + self.frame_size]))
        return out


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn accepts no connections until this returns, so the replica only goes ready
    # once the model is resident and warm.
    state["stt"] = await asyncio.to_thread(KyutaiSTT)
    state["lock"] = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/status")
async def status() -> JSONResponse:
    lock: asyncio.Lock = state["lock"]
    return JSONResponse({"ok": True, "model": MODEL_NAME, "busy": lock.locked()})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    stt: KyutaiSTT = state["stt"]
    lock: asyncio.Lock = state["lock"]
    # The decoder state is a single batch-1 stream, so a second concurrent client would
    # interleave its audio into the first one's transcript.
    if lock.locked():
        await websocket.send_text(json.dumps({"type": "busy"}))
        await websocket.close()
        return

    async with lock:
        log.info("session started")
        audio_q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        async def send_all(msgs: list[dict[str, Any]]) -> None:
            for m in msgs:
                await websocket.send_text(json.dumps(m))

        async def recv_loop() -> None:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect
                if msg.get("bytes"):
                    pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    audio_q.put_nowait(pcm)
                elif msg.get("text") and json.loads(msg["text"]).get("type") == "end":
                    audio_q.put_nowait(None)
                    return

        async def infer_loop() -> None:
            # The model expects leading silence before speech, and emits text `delay_s`
            # behind the audio, so trailing silence is what flushes the last words out.
            pending = np.zeros(int(stt.silence_prefix_s * stt.sample_rate), dtype=np.float32)
            while True:
                pcm = await audio_q.get()
                ended = pcm is None
                if ended:
                    pcm = np.zeros(int((stt.delay_s + 0.5) * stt.sample_rate), dtype=np.float32)
                pending = np.concatenate([pending, pcm])
                # Drain everything already queued so a slow step never builds a backlog
                # of tiny GPU calls.
                while not ended and not audio_q.empty():
                    nxt = audio_q.get_nowait()
                    if nxt is None:
                        ended = True
                        nxt = np.zeros(int((stt.delay_s + 0.5) * stt.sample_rate), dtype=np.float32)
                    pending = np.concatenate([pending, nxt])
                n = len(pending) // stt.frame_size * stt.frame_size
                if n:
                    frames, pending = pending[:n], pending[n:]
                    await send_all(await asyncio.to_thread(stt.step_frames, frames))
                if ended:
                    await send_all([{"type": "done"}])
                    return

        await websocket.send_text(json.dumps({"type": "ready"}))
        tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(infer_loop())]
        try:
            await asyncio.gather(*tasks)
            await websocket.close()
        except WebSocketDisconnect:
            log.info("client disconnected")
        except Exception:
            log.exception("session failed")
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.to_thread(stt.reset)
            log.info("session ended")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, ws_ping_interval=20, ws_ping_timeout=20)
