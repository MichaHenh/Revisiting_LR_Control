"""Console script for ba_parameter_free_dac."""
from __future__ import annotations

import sys
from codecarbon import track_emissions
import hydra
from ba_parameter_free_dac import run_experiment

@hydra.main(version_base=None, config_path="configs", config_name="dadaptation_libsvm")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    """Console script for ba_parameter_free_dac."""
    print(cfg)
    print(f"Configuration loaded: {cfg.name}")
    return run_experiment(cfg)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover