from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle
from nanotorch.nn import LinearLayer , L1Loss , BaseModel
from nanotorch.optimizer import SGDOptimizer
from nanotorch.tensor import Tensor
import numpy as np
from tqdm import tqdm

class Model(BaseModel):
    def __init__(self, input_size, output_size, hidden_size):
        self.layer_1 = LinearLayer(input_size,hidden_size)
        self.layer_2 = LinearLayer(hidden_size,hidden_size)
        self.layer_3 = LinearLayer(hidden_size,hidden_size)
        self.layer_4 = LinearLayer(hidden_size,output_size)
    
    def forward(self, x):
        output = self.layer_1(x).relu()
        output = self.layer_2(output).relu()
        output = self.layer_3(output).relu()
        output = self.layer_4(output)
        return output

if __name__ == "__main__":
    # Load California housing dataset
    california = fetch_california_housing()
    data_x = california.data
    data_y = california.target

    num_features = len(data_x[0])
    num_output = 1


    batch_size = 256
    num_epochs = 1000
    hidden_size = 32
    lr = 0.0001

    model = Model(num_features,num_output,hidden_size)
    optimizer = SGDOptimizer(model.parameters(), lr=lr)

    for epoch_i in range(num_epochs):
        # shuffle data
        data_x, data_y = shuffle(data_x, data_y, random_state=epoch_i)

        # estimates number of batches
        num_batches = len(data_x) // batch_size
        if len(data_x) % batch_size != 0:
            num_batches += 1

        with tqdm(total=num_batches) as pbar:
            pbar.set_description(f'Epoch #{epoch_i+1}| Batch')

            epoch_loss = 0.0
            for batch_i in range(num_batches):
                
                # formulate batch
                batch_x = Tensor(data_x[batch_i*batch_size:(batch_i+1)*batch_size])
                batch_y = Tensor(np.expand_dims(data_y[batch_i*batch_size:(batch_i+1)*batch_size],-1))

                # forward and loss
                y_pred = model(batch_x)
                loss = L1Loss()(y_pred, batch_y)

                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # log
                epoch_loss += loss.data

                pbar.update(1)
            
            # normalize losses
            epoch_loss /= num_batches

            # log
            pbar.set_description(f'Epoch #{epoch_i+1}| Epoch Loss: {epoch_loss:.2f} | Batch')