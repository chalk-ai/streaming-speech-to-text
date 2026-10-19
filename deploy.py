#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.1"]
# ///
"""Deploy Kyutai streaming STT (server.py + index.html) as a GPU ScalingGroup.

One ScalingGroup replica holds the model's streaming state and admits one WebSocket at a
time. It also serves the page, so the browser's WebSocket is same-origin.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4]
Then open the printed URL and press "Start talking".
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_NAME = "kyutai/stt-1b-en_fr"
HF_HOME = "/models/hf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="kyutai-stt")
    # Defaults to the environment in your existing `chalk login` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    from chalkcompute import Image, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(
            [
                "moshi==0.2.9",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
                "huggingface-hub==0.33.5",
            ]
        )
        .env({"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
        # Baking the weights in keeps a cold replica off Hugging Face. The layer sits before
        # the app files, so code edits reuse it.
        .run_commands(
            f'HF_HOME={HF_HOME} python -c "from huggingface_hub import snapshot_download; '
            + f"snapshot_download('{MODEL_NAME}')\""
        )
        # On CUDA, moshi torch.compiles its streaming step, and Triton builds its launcher
        # with the system C compiler at first use.
        .apt_install(["gcc", "libc6-dev"])
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="4",
        memory="16Gi",
        env={"KYUTAI_MODEL": MODEL_NAME, "PORT": "8080"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        # The model holds one batch-1 stream, so more traffic needs more replicas, and a
        # WebSocket would need sticky routing to reach the replica holding its state.
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
