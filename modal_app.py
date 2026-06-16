"""Deploy Memorae to Modal: modal deploy modal_app.py"""

from __future__ import annotations

import modal

APP_NAME = "memorae"
ROOT = "/root"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("memorae")
    .add_local_file("api.py", remote_path=f"{ROOT}/api.py")
    .add_local_file("memorae_mock_events.json", remote_path=f"{ROOT}/memorae_mock_events.json")
    .add_local_dir("web", remote_path=f"{ROOT}/web")
)

volume = modal.Volume.from_name("memorae-chroma", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("memorae-env")],
    volumes={"/data/chroma": volume},
    env={
        "CHROMA_PERSIST_DIR": "/data/chroma",
        "MEMORAE_EVENTS": f"{ROOT}/memorae_mock_events.json",
        "MEMORAE_RAG": "1",
        "MEMORAE_LLM": "1",
    },
    timeout=600,
    memory=4096,
    cpu=2.0,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import sys

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from api import app as fastapi_app

    return fastapi_app
