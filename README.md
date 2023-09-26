# lightnet
Simple educational-intended deep learning library.

> **Note**
> This library is an experimental implementation playground for me to strenghen my theory knowlodge in the field. I think it helps me or anyone
in the path of learning to have a clearer view of modules for not only a deeper grasp, but for new ideas in the manner of architectures or optimizations.

## Interface

### Structure
```py
from lightnet.models import FullyConnected
from lightnet.layers import Dense
from lightnet.losses import BinaryCrossEntropy
from lightnet.activations import Sigmoid, ReLU
```
### Craft
```py
model = FullyConnected(
    BinaryCrossEntropy(),
    [
        Dense(50, activation=ReLU()),
        Dense(4, activation=ReLU()),
        Dense(4, activation=ReLU()),
        Dense(1, activation=Sigmoid()),
    ],
)
```
### Overview
```py
print(model)
```
```
+------------------------------+
|     FullyConnected P:270     |
+-------+---------+------------+
| Layer |  Shape  | Activation |
+-------+---------+------------+
|   1   | (50, X) |    ReLU    |
|   2   | (4, 50) |    ReLU    |
|   3   |  (4, 4) |    ReLU    |
|   4   |  (1, 4) |  Sigmoid   |
+-------+---------+------------+
```
### Train
```py
losses = model.train(X_train, Y_train, epochs=1000, lr=0.001)

# plt.plot(losses)
# plt.show()
# Y_pred = model(X_test_sample)
```
### Save
```py
model.save("test.npz")
# model.load("test.npz")
```
