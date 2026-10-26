import numpy as np
import torch
import torch.nn.functional as F


class LogisticRegressionSGD:
    """Binary logistic regression trained by minibatch SGD on cross-entropy."""

    def __init__(self, lr=0.05, epochs=60, batch_size=64, l2=0.0, seed=42):
        self.lr, self.epochs, self.bs, self.l2, self.seed = lr, epochs, batch_size, l2, seed
        self.history = []

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32)
        n, d = X.shape
        self.w = torch.zeros(d, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.SGD([self.w, self.b], lr=self.lr)
        self.history = []
        for _ in range(self.epochs):
            perm = torch.randperm(n); tot = 0.0
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                loss = (F.binary_cross_entropy_with_logits(X[idx] @ self.w + self.b, y[idx])
                        + self.l2 * (self.w ** 2).sum())
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            self.history.append(tot / n)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
            return torch.sigmoid(Xt @ self.w + self.b).numpy()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
