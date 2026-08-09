import math

def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    res: list[bool] = []
    best_loss = float("inf")
    since_improvement = 0
    stopped = False

    for loss in val_losses:
        improvement = best_loss - loss
        improved = improvement > min_delta and not math.isclose(
            improvement, min_delta, rel_tol=1e-9, abs_tol=1e-12
        )
        if improved:
            best_loss = loss
            since_improvement = 0
            stopped = False          # <- the fix
        else:
            since_improvement += 1
            stopped = since_improvement >= patience
        res.append(stopped)

    return res