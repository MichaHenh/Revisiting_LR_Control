"""Console script for ba_parameter_free_dac."""
from __future__ import annotations

import sys
import hydra
from ba_parameter_free_dac import run_experiment

@hydra.main(version_base=None, config_path="configs", config_name="dadaptation_libsvm")
def main(cfg):
    """Console script for ba_parameter_free_dac."""
    print(cfg)
    print(f"Configuration loaded: {cfg.name}")
    return run_experiment(cfg)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover