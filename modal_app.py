import os
from pathlib import Path
import modal

from trailcaml import TrailCaML
from datasets.trailcamera import TrailCameraDataset

app = modal.App("trailcaml")
app.image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "pillow", "numpy", "lightning", "tensorboard"
)
vol = modal.Volume.from_name("trailcaml-data", create_if_missing=True)


@app.function(
    cpu=8.0,
    gpu="L4",
    memory=(1024 * 8),
    timeout=(300 * 6),
    volumes={"/app": vol},
)
def train(
    epochs: int,
    lr: float,
    fine_tune_after: int,
    lr_reduction: float,
    batch_size: int,
    num_workers: int,
):
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger

    torch.set_float32_matmul_precision("medium")

    data_set = TrailCameraDataset(data_dir=Path("/app/data/trailcam-dataset/"))
    train, valid, _ = data_set.dataloader_splits(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        logger=TensorBoardLogger(save_dir="/app/lightning_logs"),
        callbacks=[
            # Save best models
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                filename="{epoch}-{val_loss:.2f}",
            ),
            # Stop if not improving
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
        gradient_clip_val=0.5,
        deterministic=True,
    )

    model = TrailCaML(
        lr=lr,
        fine_tune_after=fine_tune_after,
        lr_reduction=lr_reduction,
        img_size=(240, 240),
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=valid)
    vol.commit()


@app.function(volumes={"/app": vol}, timeout=300 * 6)
@modal.web_server(6006)
def serve_tensor_board():
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "/app/lightning_logs", "--host", "0.0.0.0"])
    url = tb.launch()
    print(f"TensorBoard started at: {url}")


def upload_dataset_to_modal(
    vol: modal.Volume,
    dataset_dir: Path = Path("data/trailcam-dataset"),
    remote_dir: Path = Path("data/trailcam-dataset"),
    batch_size: int = 12,
):
    remote_dir = remote_dir
    splits = ["processed/train", "processed/valid", "processed/test"]

    def chunk(imgs, batch_size):
        while imgs:
            chunk, imgs = imgs[:batch_size], imgs[batch_size:]
            yield chunk

    for split in splits:
        print(f"Split: {split}")

        local_split = dataset_dir / split
        local_images = set(
            map(lambda f: str(dataset_dir / split / f), os.listdir(local_split))
        )

        remote_split = remote_dir / split
        try:
            remote_images = set(
                map(lambda d: str(d.path), vol.listdir(str(remote_split)))
            )
        except Exception as e:
            print(f"Remote directory not found, assuming empty: {e}")
            remote_images = set()

        print(f"Total Local images in split: {len(local_images)}")
        print(f"Total remote images in split: {len(remote_images)}")
        print(list(local_images)[:5], list(remote_images)[:5])
        images_to_upload = local_images - remote_images
        print(f"Total files to upload for split: {len(images_to_upload)}")

        for i, batch in enumerate(chunk(list(images_to_upload), batch_size)):
            print(f"Uploading Batch {i}, {i} of {len(images_to_upload) / batch_size}")
            with vol.batch_upload() as uploader:
                for img in batch:
                    uploader.put_file(
                        str(img),
                        str(img),
                    )


@app.local_entrypoint()
def main(
    upload_data: bool = False,
    train_model: bool = False,
    epochs: int = 10,
    train_batch_size: int = 32,
    lr: float = 1e-4,
    lr_reduction: float = 1e2,
    upload_batch_size: int = 12,
    fine_tune_after: int = 5,
):
    if upload_data:
        upload_dataset_to_modal(vol=vol, batch_size=upload_batch_size)
        print("Upload complete.")
    if train_model:
        train.remote(
            epochs=epochs,
            lr=lr,
            lr_reduction=lr_reduction,
            batch_size=train_batch_size,
            fine_tune_after=fine_tune_after,
        )
