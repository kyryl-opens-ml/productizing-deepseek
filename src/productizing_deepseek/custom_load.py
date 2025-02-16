import modal
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = "/productizing-deepseek"
volume = modal.Volume.from_name("productizing-deepseek", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",
            "hf-transfer",
            "python-dotenv"
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],
        revision=model_revision,
        force_download=force_download,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str,
    model_revision: Optional[str] = None,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)