import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def evaluate(
    model,
    test_loader,
    device="cuda",
    save_path="results/reconstruction_error.png",
):
    model.eval()
    errors = []

    with torch.no_grad():
        for (x,) in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            x_hat = model(x)

            # reconstruction error per window
            err = ((x_hat - x) ** 2).mean(dim=(1, 2))
            errors.append(err.cpu())

    errors = torch.cat(errors).numpy()

    # threshold (mean + 3 std)
    mean = errors.mean()
    std = errors.std()
    threshold = mean + 3 * std

    # plot
    plt.figure(figsize=(14, 4))
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold (μ + 3σ = {threshold:.3f})",
    )
    plt.xlabel("Window Index")
    plt.ylabel("MSE")
    plt.title("Reconstruction Error Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved result to {save_path}")
    print(f"Mean error: {mean:.4f}")
    print(f"Std error : {std:.4f}")
    print(f"Threshold : {threshold:.4f}")

    return errors, threshold