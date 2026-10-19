# Streaming speech-to-text on Chalk

Talk into a browser tab and watch your words appear about half a second behind you. The page and
the model are served from a single GPU replica deployed with
[`chalkcompute`](https://pypi.org/project/chalkcompute/). The model is Kyutai's
[`stt-1b-en_fr`](https://huggingface.co/kyutai/stt-1b-en_fr), a streaming speech-to-text model for
English and French.

```
browser mic ──AudioWorklet──▶ 24 kHz int16 PCM ──WebSocket──▶ GPU replica (moshi) ──▶ words ──▶ page
```

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Builds the image (moshi + torch, with the weights baked in) and deploys a `chalkcompute.ScalingGroup` on one GPU. |
| `server.py` | FastAPI app in the container: `GET /` serves the page, `WS /ws` streams audio in and text out. |
| `index.html` | Captures the microphone, resamples it to 24 kHz in an AudioWorklet, and renders words as they arrive. |
| `stream_file.py` | Streams an audio file to `/ws` in real time, for testing from the command line. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule GPU workloads.
Credentials come from the SDK’s ambient credentials from `chalk login`. You can also set `CHALK_API_SERVER`,
`CHALK_CLIENT_ID`, `CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID` yourself.

```bash
./deploy.py
```

The first deploy builds a multi-gigabyte image, then waits for a GPU node, so expect 10–20 minutes.
Later deploys reuse the cached torch and weights layers. When it finishes, the script prints a
`web_url`. Open it and press **Start talking**. The URL is HTTPS, which browsers require for
microphone access.

To test without a microphone:

```bash
./stream_file.py https://<web_url>               # real time
./stream_file.py https://<web_url> --speed 4     # 4x faster than real time
```

## WebSocket protocol

| Direction | Message |
| --- | --- |
| client → server | binary frames of 24 kHz mono little-endian int16 PCM, any size |
| client → server | `{"type": "end"}`: no more audio |
| server → client | `{"type": "ready"}`: this connection owns the model |
| server → client | `{"type": "busy"}`, then close: another stream is using the model |
| server → client | `{"type": "word", "text": " hello"}` |
| server → client | `{"type": "end_of_turn"}`: the model's voice-activity head detected the speaker stopping |
| server → client | `{"type": "done"}`: all audio after `end` has been transcribed |

The model emits text 0.5 s behind the audio. On `end`, the server feeds that much silence so the
last words are flushed out before it sends `done`.

## Notes

- **One speaker per replica.** The model keeps a single batch-1 streaming state, so the server
  admits one WebSocket at a time and answers `busy` to the rest. To serve more people, add replicas
  with sticky routing, or batch several streams through moshi's `LMGen`.
- **The image needs a C compiler.** On CUDA, moshi `torch.compile`s its streaming step, and
  Triton compiles a small launcher at first use. Without `gcc`, startup fails with
  `Failed to find C compiler`.
- **The URL is public.** `authenticated` is off so a browser can open the page directly, which
  means anyone with the link can use the GPU. Delete the group when you are done:

  ```bash
  uv run --with chalkcompute python -c \
    "from chalkcompute import ScalingGroup; ScalingGroup.from_name('kyutai-stt').delete()"
  ```

## License

The code is Apache-2.0. The `kyutai/stt-1b-en_fr` weights are released by Kyutai under
CC-BY 4.0, and the [moshi](https://github.com/kyutai-labs/moshi) Python package is MIT.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
