import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from nanotorch.tensor import Tensor
from nanotorch.nn import LinearLayer, MSELoss, RNNLayer, EarlyStopping, BaseModel
from nanotorch.optimizer import (
    SGDOptimizer,
    SGDMomentumOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    AdaGradOptimizer
)

# -------------------------------
# Dataset utilities
# -------------------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# -------------------------------
# RNN Model
# -------------------------------
class RNNModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = RNNLayer(input_size, hidden_size)
        self.fc = LinearLayer(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len, _ = x.shape()
        hidden = Tensor(np.zeros((batch_size, self.hidden_size)))

        for t in range(seq_len):
            hidden = self.rnn(x[:, t, :], hidden)

        return self.fc(hidden)

    def __call__(self, x):
        return self.forward(x)

# Training function 

def train_model(
    optimizer_name="sgd",
    lr=1e-3,
    momentum=0.9,
    beta=0.9,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    epochs=100,
    batch_size=32,
    hidden_size=64,
    seq_len=30,
    early_stopping=False,
    patience=10,
    min_delta=1e-4,
    verbose=True
):
    # -------------------------------
    # Load dataset
    # -------------------------------
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "refs/heads/master/daily-max-temperatures.csv"
    )
    df = pd.read_csv(url)
    temps = df["Temperature"].values.reshape(-1, 1)

    # Normalize
    temps = (temps - temps.min()) / (temps.max() - temps.min())

    split = int(len(temps) * 0.8)
    train_data = temps[:split]

    X_train, y_train = create_sequences(train_data, seq_len)
    # Model
    model = RNNModel(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1
    )
    # Optimizer
    if optimizer_name == "sgd":
        optimizer = SGDOptimizer(model.parameters(), lr=lr)

    elif optimizer_name == "momentum":
        optimizer = SGDMomentumOptimizer(
            model.parameters(), lr=lr, momentum=momentum
        )

    elif optimizer_name == "rmsprop":
        optimizer = RMSPropOptimizer(
            model.parameters(), lr=lr, beta=beta, eps=eps
        )

    elif optimizer_name == "adam":
        optimizer = AdamOptimizer(
            model.parameters(), lr=lr, beta1=beta1, beta2=beta2, eps=eps
        )

    elif optimizer_name == "adagrad":
        optimizer = AdaGradOptimizer(
            model.parameters(), lr=lr, eps=eps
        )

    else:
        raise ValueError("Unsupported optimizer")

    criterion = MSELoss()

    early_stopper = None
    if early_stopping:
        early_stopper = EarlyStopping(
            patience=patience,
            min_delta=min_delta
        )

    # -------------------------------
    # Training loop (tqdm FIX)
    # -------------------------------
    epoch_losses = []
    num_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(epochs):
        epoch_loss = 0.0

        pbar = tqdm(
            range(num_batches),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True
        )

        for i in pbar:
            start = i * batch_size
            end = (i + 1) * batch_size

            x = Tensor(X_train[start:end])
            y = Tensor(y_train[start:end])

            preds = model(x)
            loss = criterion(y, preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data
            pbar.set_postfix(loss=f"{loss.data:.6f}")

        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")

        if early_stopper and early_stopper(avg_loss):
            print("Early stopping triggered")
            break

    return epoch_losses

# Main (CLI)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanotorch RNN Trainer")

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "rmsprop", "adam", "adagrad"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    args = parser.parse_args()

    epoch_losses = train_model(
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta
    )
