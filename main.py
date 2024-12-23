from trailcaml import TrailCaML, train_model
from datasets import TrailCameraDataset, fetch_if_missing


def main():
    data_paths = fetch_if_missing()
    train = TrailCameraDataset(data_paths["train"])
    test = TrailCameraDataset(data_paths["test"])
    model = TrailCaML()
    train_model(model, train, test)


if __name__ == "__main__":
    main()
