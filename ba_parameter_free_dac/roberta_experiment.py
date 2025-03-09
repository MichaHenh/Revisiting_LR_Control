import os
import sys
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig
from roberta_trainer import main_worker, main

@hydra.main(version_base=None, config_path="configs", config_name="adamfixed_bookwiki_roberta")
def main_wrapper(cfg):
    if cfg.get("manual_ddp", False):
        print(f"Launching distributed run with {cfg.nproc} processes.")
        mp.spawn(main_worker, nprocs=cfg.nproc, args=(cfg,))
    else:
        # Otherwise, run in single-process (non-distributed) mode.
        main(cfg)

if __name__ == "__main__":
    sys.exit(main_wrapper())  # pragma: no cover