import numpy as np
import pandas as pd
from tqdm import tqdm

from nanotorch.tensor import Tensor
from nanotorch.nn import LinearLayer, MSELoss, RNNLayer
from nanotorch.optimizer import SGDOptimizer
import matplotlib.pyplot as plt

# -------------------------------
# Dataset utilities
# -------------------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# -------------------------------
# RNN Model
# -------------------------------
class RNNModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn = RNNLayer(input_size, hidden_size)
        self.fc = LinearLayer(hidden_size, output_size)
        self.hidden_size = hidden_size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape()
        hidden_state = Tensor(np.zeros((batch_size, self.hidden_size)))
        for t in range(seq_len):
            hidden_state = self.rnn(x[:,t,:], hidden_state)
        return self.fc(hidden_state)

    def parameters(self):
        return [self.rnn, self.fc]

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    # Load Sydney temperature dataset
    dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/daily-max-temperatures.csv"
    df = pd.read_csv(dataset_url)
    temps = df["Temperature"].values.reshape(-1, 1)

    # Normalize
    temps = (temps - temps.min()) / (temps.max() - temps.min())

    # Hyperparameters
    seq_len = 30
    batch_size = 32
    num_epochs = 100
    hidden_size = 64
    lr = 0.001

    # Train / test split
    split = int(len(temps) * 0.8)
    train_data = temps[:split]
    test_data = temps[split - seq_len :]

    X_train, y_train = create_sequences(train_data, seq_len)
    X_test, y_test = create_sequences(test_data, seq_len)

    # Model, optimizer, loss
    model = RNNModel(input_size=1, hidden_size=hidden_size, output_size=1)
    optimizer = SGDOptimizer(model.parameters(), lr=lr)
    criterion = MSELoss()

    # -------------------------------
    # Training loop
    # -------------------------------
    epoch_losses = []
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        num_batches = len(X_train) // batch_size
        if len(X_train) % batch_size != 0:
            num_batches += 1

        with tqdm(total=num_batches) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            for batch_i in range(num_batches):

                batch_x = Tensor(
                    X_train[batch_i * batch_size : (batch_i + 1) * batch_size]
                )
                batch_y = Tensor(
                    y_train[batch_i * batch_size : (batch_i + 1) * batch_size]
                )

                # Forward
                preds = model(batch_x)
                loss = criterion(batch_y, preds)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.data
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.data:.6f}")

        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {epoch_loss:.6f}")


    for category_name, category_data in [("Train", epoch_losses)]:
        plt.figure(figsize=(8, 5))
        plt.plot(category_data, label="RNN model")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(category_name + " Loss Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(category_name.lower() + "_loss_comparison.png")
        plt.close()
