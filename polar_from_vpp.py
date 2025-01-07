from pathlib import Path

import pinta.settings as settings
from pinta.data_processing.load import load_folder, load_sets
from pinta.data_processing.plot import polar_plot
from pinta.data_processing.training_set import TrainingSetBundle
from pinta.model.model_factory import model_factory
from pinta.synthetic import polar
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="mini_polar")
def run(cfg: DictConfig):
    """
    Load a given engine, generate a couple of synthetic plots from it
    """

    # a bit hacky: get the normalization factors on the fly
    print(cfg)
    data = load_folder(
        Path(cfg.data.path),
        zero_mean_helm=False,
        max_number_sequences=cfg.data.get("max_number_sequences", -1),
    )
    datasplits = load_sets(data, cfg)
    mean, std = TrainingSetBundle(datasplits).get_norm()

    model = model_factory(cfg, model_path=cfg.model.path)
    model = model.to(device=settings.device)

    if not model.valid:
        print("Failed loading the model, cannot continue")
        exit(-1)

    # Generate data all along the curve
    polar_data = polar.generate(
        engine=model,
        wind_range=[5, 25],
        wind_step=5,
        angular_step=0.1,
        seq_len=cfg.model.seq_length,
        mean=mean,
        std=std,
        inputs=cfg.inputs,
    )

    # Plot all that stuff
    polar_plot(polar_data)


if __name__ == "__main__":
    run()
