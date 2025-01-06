# TrailCaML

TrailCaML helps identify animals in trail camera images.

## Deployment

The model and dataset are large enough that a GPU is neccessary to
train the model in a reasonable amount of time. `deploy.py` defines a
Modal App that handles training and storage of checkpoints and metrics.

To deploy an updated app:
```
uv run modal deploy deploy.py
```

The Modal App also defines a web endpoint serving tensorboard.

## Training

The model is deployed via a Modal App that has access to enterprise GPUs.

To start a new training run:
```ipython
import modal

train = modal.Function.lookup("trailcaml", "train")

with modal.enable_output():
  job = train.remote(epochs=16, lr=1e-4, fine_tune_after=8)
```

See the tensorboard endpoint to review the training metrics.
