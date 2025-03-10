import os
import sys
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig
from roberta_trainer import main_worker, main
import socket

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
        # os.chdir('.')
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = free_port

        print(f"Launching distributed run with {cfg.nproc} processes.")
        # mp.set_start_method("spawn", force=True)
        result = mp.spawn(main_worker, nprocs=cfg.nproc, args=(cfg,), join=True)
        print(result)
        return result
    else:
        # Otherwise, run in single-process (non-distributed) mode.
        return main(cfg)

if __name__ == "__main__":
    sys.exit(main_wrapper())  # pragma: no cover