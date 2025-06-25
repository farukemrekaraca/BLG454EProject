import os
import copy
import time
import datetime as _dt
from typing import Tuple, Optional

import torch
import numpy as np
import torchvision.utils as vutils

from utils.common import (
    DatasetInfo,
    TensorDataset,
    iteration,
    get_network,
    get_current_time,
    get_match_loss,
    get_random_images,
    get_outer_and_inner_loops,
)
from utils.consts import (
    DEVICE,
    ITERATIONS,
    BATCH_SIZE_REAL,
    BATCH_SIZE_TRAIN,
)

__all__ = ["distill_synthetic_dataset"]

# ----------------------------------------------------------------------
# Helper ----------------------------------------------------------------
# ----------------------------------------------------------------------

def _nowstamp() -> str:
    """Filesystem‑safe timestamp like 20250625‑100428"""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------
# Public API ------------------------------------------------------------
# ----------------------------------------------------------------------

def distill_synthetic_dataset(
    *,
    dataset: str = "MNIST",
    network: str = "MLP",
    ipc: int = 1,
    iterations: int = ITERATIONS,
    num_experiments: int = 1,
    syn_init: str = "random",  # "random" | "real"
    net_lr: float = 1e-2,
    syn_lr: float = 1e-1,
    save_dir: str = "syndata",
    visualize: bool = True,
    verbose: bool = True,
    log_to_file: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient‑matching dataset distillation with optional logging.
    """

    # ------------------------------------------------------------------
    # 0. I/O setup -----------------------------------------------------
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    log_fh = None
    if log_to_file:
        log_path = os.path.join(
            save_dir, f"{dataset}_{network}_ipc-{ipc}_{_nowstamp()}.log"
        )
        log_fh = open(log_path, "w", encoding="utf-8")

    def _log(msg: str):
        if verbose:
            print(msg)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Dataset & bookkeeping
    # ------------------------------------------------------------------
    dataset_info = DatasetInfo(dataset_name=dataset)
    num_classes = dataset_info.num_of_classes
    outer_loop, inner_loop = get_outer_and_inner_loops(ipc)

    _log(
        f"[distill] {dataset} | {network} | ipc={ipc} | iters={iterations} | device={DEVICE}"
    )

    # Pre‑load real data ------------------------------------------------
    real_imgs = torch.stack([d[0] for d in dataset_info.train_dataset], dim=0)  # CPU
    real_labels = torch.tensor([d[1] for d in dataset_info.train_dataset], dtype=torch.long)  # CPU
    class_indices = [torch.where(real_labels == c)[0] for c in range(num_classes)]

    last_syn: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # 2. Distillation experiments
    # ------------------------------------------------------------------
    for exp in range(num_experiments):
        _log(f"\n[distill] Experiment {exp + 1}/{num_experiments}")

        # Synthetic images & labels ------------------------------------
        syn_imgs = torch.randn(
            size=(num_classes * ipc, dataset_info.num_of_channels, *dataset_info.img_size),
            dtype=torch.float,
            device=DEVICE,
            requires_grad=True,
        )
        syn_labels = torch.arange(num_classes, device=DEVICE).repeat_interleave(ipc)

        # Optional real initialisation ---------------------------------
        if syn_init == "real":
            for c in range(num_classes):
                syn_imgs.data[c * ipc : (c + 1) * ipc] = get_random_images(
                    real_imgs, class_indices, c, ipc
                ).detach()

        img_opt = torch.optim.SGD([syn_imgs], lr=syn_lr, momentum=0.5)
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

        # --------------------------------------------------------------
        # 3. Outer optimisation loop
        # --------------------------------------------------------------
        for it in range(iterations + 1):
            net = get_network(
                network,
                dataset_info.num_of_channels,
                num_classes,
                dataset_info.img_size,
            )
            net.train()
            net_params = list(net.parameters())
            net_opt = torch.optim.SGD(net.parameters(), lr=net_lr, momentum=0.5)

            loss_acc = 0.0
            for outer in range(outer_loop):
                loss = torch.tensor(0.0, device=DEVICE)

                for c in range(num_classes):
                    real_img = get_random_images(real_imgs, class_indices, c, BATCH_SIZE_REAL).to(DEVICE)
                    real_lab = torch.full((real_img.size(0),), c, dtype=torch.long, device=DEVICE)

                    syn_img = syn_imgs[c * ipc : (c + 1) * ipc]
                    syn_lab = torch.full((ipc,), c, dtype=torch.long, device=DEVICE)

                    real_loss = loss_fn(net(real_img), real_lab)
                    g_real = [g.detach() for g in torch.autograd.grad(real_loss, net_params)]

                    syn_loss = loss_fn(net(syn_img), syn_lab)
                    g_syn = torch.autograd.grad(syn_loss, net_params, create_graph=True)

                    loss += get_match_loss(g_syn, g_real)

                img_opt.zero_grad()
                loss.backward()
                img_opt.step()
                loss_acc += loss.item()

                if outer == outer_loop - 1:
                    break

                syn_ds = TensorDataset(syn_imgs.detach(), syn_labels.detach())
                train_loader = torch.utils.data.DataLoader(
                    syn_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True
                )
                iteration(net, loss_fn, net_opt, train_loader, is_training=True)

            if it % 10 == 0:
                avg_loss = loss_acc / (num_classes * outer_loop)
                _log(
                    f"  iter {it:4d}/{iterations} | loss {avg_loss:8.4f} | {get_current_time()}"
                )

        # --------------------------------------------------------------
        # 4. Persist results for this experiment
        # --------------------------------------------------------------
        syn_imgs_cpu = syn_imgs.detach().cpu()
        syn_labels_cpu = syn_labels.detach().cpu()
        last_syn = (syn_imgs_cpu, syn_labels_cpu)

        ckpt_path = os.path.join(save_dir, f"{dataset}_{network}_ipc-{ipc}_exp-{exp}.pt")
        torch.save({"syn_imgs": syn_imgs_cpu, "syn_labels": syn_labels_cpu}, ckpt_path)

        if visualize:
            png_path = ckpt_path.replace(".pt", ".png")
            vutils.save_image(syn_imgs_cpu, png_path, nrow=ipc, normalize=True, value_range=(0, 1))

        _log(f"[distill] Saved -> {ckpt_path}{' & grid' if visualize else ''}")

    # ------------------------------------------------------------------
    # 5. Wrap‑up -------------------------------------------------------
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    _log(f"\n[distill] Total runtime: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    if log_fh is not None:
        log_fh.close()

    return last_syn  # (images, labels)


# ----------------------------------------------------------------------
# CLI ------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser("Dataset distillation via gradient matching")
    cli.add_argument("--dataset", type=str, default="MNIST")
    cli.add_argument("--network", type=str, default="MLP")
    cli.add_argument("--ipc", type=int, default=1)
    cli.add_argument("--iterations", type=int, default=ITERATIONS)
    cli.add_argument("--num_experiments", type=int, default=1)
    cli.add_argument("--syn_init", type=str, choices=["random", "real"], default="random")
    cli.add_argument("--net_lr", type=float, default=1e-2)
    cli.add_argument("--syn_lr", type=float, default=1e-1)
    cli.add_argument("--save_dir", type=str, default="syndata")
    cli.add_argument("--no_grid", action="store_true", help="Skip PNG grid save")
    cli.add_argument("--no_verbose", action="store_true", help="Silence stdout prints")
    args = cli.parse_args()
    distill_synthetic_dataset(
        dataset=args.dataset,
        network=args.network,
        ipc=args.ipc,
        iterations=args.iterations,
        num_experiments=args.num_experiments,
        syn_init=args.syn_init,
        net_lr=args.net_lr,
        syn_lr=args.syn_lr,
        save_dir=args.save_dir,
        visualize=not args.no_grid,
        verbose=not args.no_verbose,
    )
