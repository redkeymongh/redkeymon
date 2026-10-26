import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

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
class CauchyRegression:
    """Linear regression y = w0 + w.x fitted by minimizing the Cauchy loss
    l(yhat, y) = c^2/2 * log(1 + ((y - yhat)/c)^2) with Adam in PyTorch."""

    def __init__(self, c=1.0, lr=0.1, epochs=3000, seed=0):
        self.c, self.lr, self.epochs, self.seed = c, lr, epochs, seed
        self.history = []

    def _loss(self, yhat, y, reduction="mean"):
        r = (y - yhat) / self.c
        l = (self.c ** 2 / 2) * torch.log1p(r ** 2)
        return l.mean() if reduction == "mean" else l.sum()

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.X_, self.y_ = X, y
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.tensor(float(y.mean()), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=self.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1 / self.epochs))
        self.history = []
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = self._loss(Xt @ w + b, yt)
            loss.backward(); opt.step(); sched.step()
            self.history.append(loss.item())
        w_s, b_s = w.detach().numpy(), b.item()
        self.coef_ = w_s / self.sd_
        self.intercept_ = b_s - np.sum(w_s * self.mu_ / self.sd_)
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, float) @ self.coef_

    def loss(self, X=None, y=None, reduction="mean"):
        X = self.X_ if X is None else X
        y = self.y_ if y is None else y
        return self._loss(torch.tensor(self.predict(X)),
                          torch.tensor(np.asarray(y, float)), reduction).item()

    def conf_int(self, alpha=0.05):
        """Asymptotic sandwich (M-estimator) confidence intervals."""
        n = len(self.y_); Xd = np.column_stack([np.ones(n), self.X_])
        theta = np.r_[self.intercept_, self.coef_]
        r = self.y_ - Xd @ theta; c = self.c
        psi = r / (1 + (r / c) ** 2)
        dpsi = (1 - (r / c) ** 2) / (1 + (r / c) ** 2) ** 2
        H = (Xd * dpsi[:, None]).T @ Xd
        B = (Xd * (psi ** 2)[:, None]).T @ Xd
        Hi = np.linalg.inv(H)
        se = np.sqrt(np.diag(Hi @ B @ Hi)); z = norm.ppf(1 - alpha / 2)
        return pd.DataFrame({"coef": theta, "std_err": se,
                             "ci_low": theta - z * se, "ci_high": theta + z * se})
