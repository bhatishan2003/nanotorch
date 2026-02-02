import argparse
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle

from nanotorch.nn import LinearLayer, L1Loss, EarlyStopping, BaseModel
from nanotorch.optimizer import SGDOptimizer, SGDMomentumOptimizer, RMSPropOptimizer , AdamOptimizer, AdaGradOptimizer
from nanotorch.tensor import Tensor

class Model(BaseModel):
    def __init__(self, input_size, output_size, hidden_size):
        self.layer_1 = LinearLayer(input_size, hidden_size)
        self.layer_2 = LinearLayer(hidden_size, hidden_size)
        self.layer_3 = LinearLayer(hidden_size, hidden_size)
        self.layer_4 = LinearLayer(hidden_size, output_size)

    def forward(self, x):
        x = self.layer_1(x).relu()
        x = self.layer_2(x).relu()
        x = self.layer_3(x).relu()
        x = self.layer_4(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanotorch Training Script")

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum","rmsprop","adam", "adagrad"],
                        help="Optimizer type")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum (only for momentum optimizer)")

    parser.add_argument("--beta", type=float, default=0.9,
                        help="RMSProp decay rate")
    
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="RMSProp epsilon")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")

    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Hidden layer size")
    
    parser.add_argument("--beta1", type=float, default=0.9,
                    help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                    help="Adam beta2")


    args = parser.parse_args()

    # -------------------------------
    # Data
    # -------------------------------
    california = fetch_california_housing()
    data_x = california.data
    data_y = california.target

    num_features = data_x.shape[1]
    num_output = 1

    # -------------------------------
    # Model
    # -------------------------------
    model = Model(num_features, num_output, args.hidden_size)

    # -------------------------------
    # Optimizer selection
    # -------------------------------
    if args.optimizer == "sgd":
        optimizer = SGDOptimizer(
            model.parameters(),
            lr=args.lr
        )
    elif args.optimizer == "momentum":
        optimizer = SGDMomentumOptimizer(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum
        )
    elif args.optimizer == "rmsprop":
        optimizer = RMSPropOptimizer(
            model.parameters(),
            lr=args.lr,
            beta=args.beta,
            eps=args.eps
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
            model.parameters(),
            lr=args.lr,
            eps=args.eps
        )

    else:
        raise ValueError("Unsupported optimizer")

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(args.epochs):

        data_x, data_y = shuffle(data_x, data_y, random_state=epoch)

        num_batches = len(data_x) // args.batch_size
        if len(data_x) % args.batch_size != 0:
            num_batches += 1

        epoch_loss = 0.0

        with tqdm(total=num_batches) as pbar:
            pbar.set_description(f"Epoch {epoch+1}")

            for batch_i in range(num_batches):

                start = batch_i * args.batch_size
                end = (batch_i + 1) * args.batch_size

                batch_x = Tensor(data_x[start:end])
                batch_y = Tensor(
                    np.expand_dims(data_y[start:end], axis=-1)
                )

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
