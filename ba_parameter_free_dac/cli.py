"""Console script for ba_parameter_free_dac."""
from __future__ import annotations

import sys

import memray
from codecarbon import track_emissions
import hydra

from ba_parameter_free_dac import cool_things
from experiment import run

@hydra.main(version_base=None, config_path="configs", config_name="adam_fixed_cifar10")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    """Console script for ba_parameter_free_dac."""
    with memray.Tracker("memray.bin"):
        print(f"Hello, I am a test! My name is {cfg.name}")
        run(cfg)

        return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover