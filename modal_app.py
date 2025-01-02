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


@app.function(cpu=8.0, gpu="L4", volumes={"/app": vol}, timeout=300 * 6)
def train(epochs: int):
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    torch.set_float32_matmul_precision("medium")

    data_set = TrailCameraDataset(data_dir=Path("/app/data/trailcam-dataset/"))
    train, valid, test = data_set.dataloader_splits(batch_size=32, num_workers=8)

    logger = L.pytorch.loggers.TensorBoardLogger(save_dir="/app/lightning_logs")
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=6,
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

    model = TrailCaML(lr=1e-3, fine_tune_after=5, img_size=(240, 240))
    trainer.fit(model, train_dataloaders=train, val_dataloaders=valid)
    vol.commit()


@app.function(volumes={"/app": vol})
@modal.web_server(6006)
def serve_tensor_board():
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "/app/lightning_logs", "--host", "0.0.0.0"])
    url = tb.launch()
    print(f"TensorBoard started at: {url}")


def upload_dataset_to_modal(
    vol: modal.Volume,
    mnt_dir: Path = Path("/app"),
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

        print(f"Total Local images: {len(local_images)}")
        print(f"Total remote images: {len(remote_images)}")
        print(list(local_images)[:5], list(remote_images)[:5])
        images_to_upload = local_images - remote_images
        print(f"Total files to upload: {len(images_to_upload)}")

        for i, batch in enumerate(chunk(list(images_to_upload), batch_size)):
            print(f"Uploading Batch: {i}...")
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
    upload_batch_size: int = 12,
):
    if upload_data:
        upload_dataset_to_modal(vol=vol, batch_size=upload_batch_size)
        print("Upload complete.")
    if train_model:
        train.remote(epochs)
