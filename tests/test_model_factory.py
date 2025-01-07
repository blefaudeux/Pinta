from pinta.model.model_factory import model_factory
from pinta.settings import ModelType
from omegaconf import OmegaConf


def test_create_models():
    settings = {
        "inputs": ["tws", "twa_x", "twa_y", "heel"],
        "outputs": ["sog"],
        "transforms": [
            ("normalize", []),
            ("random_flip", [["heel", "twa_y"], 0.5]),
            ("single_precision", []),
        ],
        "tuning_inputs": [],
        "model": {
            "model_type": "mlp",
            "dropout": 0.25,
            "inner_layers": 3,
            "gru_layers": 2,
            "kernel_sizes": [3, 3],
            "hidden_size": 256,
            "seq_length": 27,
        },
        "data": {
            "training_ratio": 0.9,
            "train_batch_size": 100000,
            "test_batch_size": 10000,
        },
        "epoch": 15,
        "optim": {
            "name": "adamw",
            "learning_rate": 0.01,
        },
        "log": "test",
        "scheduler": "reduce_plateau",
        "amp": False,
    }

    for m_type in ModelType:
        settings["model_type"] = m_type.value

        cfg = OmegaConf.create(settings)
        _ = model_factory(cfg, model_path="")


if __name__ == "__main__":
    test_create_models()
