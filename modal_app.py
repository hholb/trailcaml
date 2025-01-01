import modal
import torch
from pathlib import Path
from trailcaml import TrailCaML
from datasets.trailcamera import TrailCameraDataset, fetch_if_missing

app = modal.App("trailcaml")
app.image = modal.Image.debian_slim().pip_install(
    "torch", "pillow", "numpy", "polars", "lightning", "tensorboard"
)
vol = modal.Volume.from_name("trailcaml-data", create_if_missing=True)


@app.function(gpu="L4", volumes={"/root/data": vol}, timeout=600)
def train(epochs: int):
    import torch
    import lightning as L
    from torch.utils.data import DataLoader

    torch.set_float32_matmul_precision("medium")

    data_paths = fetch_if_missing()
    train_data = DataLoader(TrailCameraDataset(data_paths["train"]), batch_size=32)
    val_data = DataLoader(
        TrailCameraDataset(data_paths["validation"]), batch_size=32, num_workers=4
    )
    model = TrailCaML()

    logger = L.pytorch.loggers.TensorBoardLogger("/root/data/lightning_logs")
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        default_root_dir="/root/data/lightning_logs",
        log_every_n_steps=32,
    )
    trainer.fit(model, train_data)
    trainer.validate(model, val_data)

    vol.commit()


@app.function(volumes={"/root/data": vol})
@modal.web_server(6006)
def serve_tensor_board():
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(
        argv=[None, "--logdir", "/root/data/lightning_logs", "--host", "0.0.0.0"]
    )
    url = tb.launch()
    print(f"TensorBoard started at: {url}")


@app.local_entrypoint()
def main(
    epochs: int = 10,
):
    model_path_remote = train.remote(epochs)

    if model_path_remote is None:
        print("Training failed")
        return
    else:
        print(f"Training complete! Model saved to: {model_path_remote}")
