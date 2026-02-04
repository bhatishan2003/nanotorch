import argparse
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.utils import shuffle

from nanotorch.nn import Conv2DLayer, LinearLayer, L1Loss, BaseModel

from nanotorch.optimizer import (
    SGDOptimizer,
    SGDMomentumOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    AdaGradOptimizer
)
from nanotorch.tensor import Tensor


# -------------------------------------------------
# CNN Model
# -------------------------------------------------
class CNNModel(BaseModel):
    def __init__(self, num_classes=10):
        # Input: (B, 1, 8, 8)
        self.conv1 = Conv2DLayer(1, 4, kernel_size=3)   # → (B, 4, 6, 6)

        # Flattened size = 4 * 6 * 6 = 144
        self.fc1 = LinearLayer(144, 64)
        self.fc2 = LinearLayer(64, num_classes)

    def forward(self, x):
        x = self.conv1(x).relu()

        # Flatten
        B = x.shape()[0]
        x = Tensor(x.data.reshape(B, -1))

        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x
# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanotorch CNN Training")

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "rmsprop", "adam", "adagrad"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    digits = load_digits()
    data_x = digits.images        # (N, 8, 8)
    data_y = digits.target        # (N,)

    # Normalize & add channel dim
    data_x = data_x / 16.0
    data_x = np.expand_dims(data_x, axis=1)  # (N, 1, 8, 8)

    num_classes = 10

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = CNNModel(num_classes=num_classes)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------
    if args.optimizer == "sgd":
        optimizer = SGDOptimizer(model.parameters(), lr=args.lr)

    elif args.optimizer == "momentum":
        optimizer = SGDMomentumOptimizer(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )

    elif args.optimizer == "rmsprop":
        optimizer = RMSPropOptimizer(
            model.parameters(), lr=args.lr, beta=args.beta, eps=args.eps
        )

    elif args.optimizer == "adam":
        optimizer = AdamOptimizer(
            model.parameters(),
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps
        )

    elif args.optimizer == "adagrad":
        optimizer = AdaGradOptimizer(
            model.parameters(), lr=args.lr, eps=args.eps
        )

    else:
        raise ValueError("Unsupported optimizer")

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    for epoch in range(args.epochs):

        data_x, data_y = shuffle(data_x, data_y, random_state=epoch)

        num_batches = len(data_x) // args.batch_size
        if len(data_x) % args.batch_size != 0:
            num_batches += 1

        epoch_loss = 0.0

        with tqdm(total=num_batches) as pbar:
            pbar.set_description(f"Epoch {epoch+1}")

            for i in range(num_batches):
                start = i * args.batch_size
                end = (i + 1) * args.batch_size

                batch_x = Tensor(data_x[start:end])

                # One-hot targets
                batch_y = np.zeros((len(data_y[start:end]), num_classes))
                batch_y[np.arange(len(batch_y)), data_y[start:end]] = 1
                batch_y = Tensor(batch_y)

                # Forward
                y_pred = model(batch_x)
                loss = L1Loss()(y_pred, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.data
                pbar.update(1)

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
