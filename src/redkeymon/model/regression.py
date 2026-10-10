import numpy as np
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr, self.epochs = learning_rate, epochs
        self.w0 = torch.zeros(1, requires_grad=True)
        self.w1 = torch.zeros(1, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.w0_history, self.w1_history, self.loss_history = [], [], []

    def forward(self, x):
        return self.w0 + self.w1 * x

    def fit(self, x_train, y_train, x_test, y_test):
        x = torch.as_tensor(np.asarray(x_train), dtype=torch.float32).flatten()
        y = torch.as_tensor(np.asarray(y_train), dtype=torch.float32).flatten()
        self._x, self._y = x, y
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.forward(x), y)
            loss.backward()
            self.optimizer.step()
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())
        xt = torch.as_tensor(np.asarray(x_test), dtype=torch.float32).flatten()
        yt = torch.as_tensor(np.asarray(y_test), dtype=torch.float32).flatten()
        with torch.no_grad():
            pred = self.forward(xt)
            r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        self.r2 = r2.item()
        print(f"Test R^2: {self.r2:.4f}")
        return self

    def predict(self, x_new):
        x = torch.as_tensor(np.asarray(x_new), dtype=torch.float32).flatten()
        with torch.no_grad():
            return self.forward(x).numpy()

    def analysis_plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        x, y = self._x.numpy(), self._y.numpy()
        ax[0, 0].scatter(x, y, s=10); ax[0, 0].set_title("Original data")
        xs = np.linspace(x.min(), x.max(), 100)
        ax[0, 1].scatter(x, y, s=10)
        ax[0, 1].plot(xs, self.predict(xs), "r"); ax[0, 1].set_title("Fitted line")
        ax[1, 0].plot(self.w0_history, label="w0")
        ax[1, 0].plot(self.w1_history, label="w1")
        ax[1, 0].legend(); ax[1, 0].set_title("Parameters vs epoch")
        ax[1, 1].plot(self.loss_history); ax[1, 1].set_title("MSE loss vs epoch")
        plt.tight_layout(); plt.show()
