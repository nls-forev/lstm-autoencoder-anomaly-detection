import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import LSTMAutoencoder

root_dir = Path(
    "./dataset"
)
txt_file = root_dir / "household_power_consumption.txt"
df = pd.read_csv(txt_file, sep=";", na_values="?", low_memory=False)

series = df["Global_active_power"].astype(float).dropna().values
series = (series - series.mean()) / series.std()


def make_windows(series, T):
    X = []
    for i in range(len(series) - T):
        X.append(series[i : i + T])
    return torch.tensor(X).float().unsqueeze(-1)


T = 20
x_t = make_windows(series, T)

x_t_train = x_t[: int(0.8 * len(x_t))]
x_t_test = x_t[int(0.8 * len(x_t)) :]


train_dataset = TensorDataset(x_t_train)
test_dataset = TensorDataset(x_t_test)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    prefetch_factor=8,
    persistent_workers=True,
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8,
    prefetch_factor=8,
    persistent_workers=True,
    pin_memory=True,
)


model = LSTMAutoencoder(features=1, hidden=64, T=T, num_layers=2).to("cuda")


def train(epochs, lr):
    model.train()

    optim = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-7)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for (x,) in pbar:
            x = x.to("cuda")

            x_hat = model(x)
            loss = loss_fn(x_hat, x)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running_loss += loss.item() * x.size(0)
            n += x.size(0)

            pbar.set_postfix(loss=loss.item(), avg=running_loss / n)

        scheduler.step()


if __name__ == "__main__":
    train(epochs=10, lr=1e-3)
