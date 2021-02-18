from pinta.model.model_factory import model_factory
from pinta.settings import ModelType


def test_create_models():
    settings = {
        "inputs": ["tws", "twa_x", "twa_y", "heel"],
        "outputs": ["sog"],
        "transforms": [["normalize", []], ["random_flip", [["heel", "twa_y"], 0.5]], ["single_precision", []]],
        "model_type": "mlp",
        "hidden_size": 256,
        "seq_length": 27,
        "data": {
            "training_ratio": 0.9,
            "train_batch_size": 100000,
            "test_batch_size": 10000,
        },
        "epoch": 15,
        "learning_rate": 0.01,
        "dilated_conv": {"dropout": 0.25},
        "mlp": {"inner_layers": 3},
        "rnn": {"gru_layers": 2, "kernel_sizes": [3, 3]},
        "conv": {"kernel_sizes": [3, 3]},
        "log": "test",
        "scheduler": "reduce_plateau",
        "amp": False,
    }

    for m_type in ModelType:
        settings["model_type"] = m_type.value
        _ = model_factory(settings, model_path="")
