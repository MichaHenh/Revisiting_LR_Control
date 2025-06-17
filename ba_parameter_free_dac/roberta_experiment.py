import os
import sys
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig
from roberta_trainer import main_worker, main
import socket
from multiprocessing import Manager

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

@hydra.main(version_base=None, config_path="configs", config_name="adamfixed_bookwiki_roberta")
def main_wrapper(cfg):
    if cfg.get("manual_ddp", False):
        free_port = str(get_free_port())
        
        manager = Manager()
        return_dict = manager.dict()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = free_port

        print(f"Launching distributed run with {cfg.nproc} processes.")
        # mp.set_start_method("spawn", force=True)
        mp.spawn(main_worker, nprocs=cfg.nproc, args=(cfg, return_dict), join=True)
        print(return_dict)
        return return_dict[0]
    else:
        # Otherwise, run in single-process (non-distributed) mode.
        return main(cfg)

if __name__ == "__main__":
    sys.exit(main_wrapper())  # pragma: no cover
