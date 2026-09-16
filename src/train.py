"""Training loop: warmup + cosine schedule, per-step diagnostic hook, SAM two-pass step."""

import copy
import csv

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, eta_min=0.0, T_max=epochs)
    warmup = LinearLR(optimizer, start_factor=1.0 / warmup_epochs,
                      end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, eta_min=0.0,
                               T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def _log_epoch(epoch: int, epochs: int) -> bool:
    """Print the first epoch, the last epoch, and every 10th in between."""
    return epoch == 1 or epoch == epochs or epoch % 10 == 0


def _mean_records(records: list) -> dict:
    """Mean over a list of per-step diagnostic dicts (keys assumed consistent)."""
    if not records:
        return {}
    keys = records[0].keys()
    n = len(records)
    return {k: float(sum(r[k] for r in records)) / n for k in keys}


def train_model(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loader,
    epochs: int = 100,
    warmup_epochs: int = 5,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    patience: int = 9999,
    diag_hook=None,
    perstep_csv: str = None) -> tuple:
    """Train model and return (model, history). diag_hook(optimizer)->dict logs per-step means into history, and per-row into perstep_csv if given."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = _build_scheduler(optimizer, epochs, warmup_epochs)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    is_sam = bool(getattr(optimizer, "is_sam", False))
    no_improve = 0
    history = []
    global_step = 0
    ps_file, ps_writer = None, None
    if perstep_csv is not None:
        ps_file = open(perstep_csv, "w", newline="")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_n = 0.0, 0
        epoch_diag = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if is_sam:
                def closure():
                    optimizer.zero_grad()
                    loss_c = criterion(model(x), y)
                    loss_c.backward()
                    return loss_c
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            global_step += 1
            if diag_hook is not None:
                rec = diag_hook(optimizer)
                if rec:
                    epoch_diag.append(rec)
                    if ps_file is not None:
                        row = {"epoch": epoch, "global_step": global_step, **rec}
                        if ps_writer is None:
                            ps_writer = csv.DictWriter(ps_file, fieldnames=list(row))
                            ps_writer.writeheader()
                        ps_writer.writerow(row)
                        ps_file.flush()

            train_loss += loss.item() * len(y)
            train_n += len(y)

        scheduler.step()

        val_loss = _eval_loss(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss / train_n, "val_loss": val_loss}
        row.update(_mean_records(epoch_diag))
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"early stop epoch {epoch}, best={best_val_loss:.4f}")
                break

        if verbose and _log_epoch(epoch, epochs):
            print(f"epoch {epoch}/{epochs} train={train_loss / train_n:.4f} "
                  f"val={val_loss:.4f} best={best_val_loss:.4f}")

    if ps_file is not None:
        ps_file.close()
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        total_n += len(y)
    return total_loss / total_n
