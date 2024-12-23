import modal
import torch
from pathlib import Path
from trailcaml import TrailCaML, train_model
from datasets.trailcamera import TrailCameraDataset, fetch_if_missing

app = modal.App("trailcaml")
app.image = modal.Image.debian_slim().pip_install("torch", "pillow", "numpy", "polars")
vol = modal.Volume.from_name("trailcaml-data", create_if_missing=True)


@app.function(gpu="L4", volumes={"/root/data": vol}, timeout=600)
def train(reload_from_checkpoint=True, learning_rate: float = 0.01, epochs: int = 10):
    import torch
    MODEL_PATH = Path.home() / "data/model.pt"

    try:
        model = TrailCaML()
        if reload_from_checkpoint and MODEL_PATH.exists():
            print("Reloading model from checkpoint...")
            checkpoint = torch.load(MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])

        data_paths = fetch_if_missing()
        train = TrailCameraDataset(data_paths["train"])
        test = TrailCameraDataset(data_paths["test"])

        train_model(model, train, test, epochs=epochs, learning_rate=learning_rate)

        # Serialize the model state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model.config if hasattr(model, "config") else None,
                "model_class": model.__class__.__name__,
            },
            MODEL_PATH,
        )

        # Return the path where the model was saved
        return str(MODEL_PATH)

    except Exception as e:
        import traceback

        traceback.print_tb(e.__traceback__)
        print(f"Error during training: {e}")
        return None
    finally:
        # Ensure volume changes are persisted
        vol.commit()


@app.local_entrypoint()
def main(
    reload_from_checkpoint: bool = False,
    learning_rate: float = 0.01,
    epochs: int = 10,
):
    model_path_remote = train.remote(reload_from_checkpoint, learning_rate, epochs)

    if model_path_remote is None:
        print("Training failed")
        return
    else:
        print(f"Training complete! Model saved to: {model_path_remote}")
