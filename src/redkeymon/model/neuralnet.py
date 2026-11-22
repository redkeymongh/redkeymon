"""
neuralnet.py — neural network models for the package's `model` subpackage.

Usage:
    from redkeymon.model.neuralnet import TorchNet
    model = TorchNet(input_dim=8, output_dim=1)
    model.fit(X_train, y_train, epochs=1000)
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler


class TorchNet(nn.Module):
    """Feed-forward classifier: input_dim -> 4 -> 3 -> output_dim (ReLU hidden layers)."""

    def __init__(self, input_dim=8, output_dim=1, hidden=(4, 3), lr=0.01, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], output_dim))
        # BCEWithLogits for binary (output_dim=1); CrossEntropy for multi-class
        self.criterion = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scaler, self.losses = None, []

    # 1. Data standardization
    def standardize(self, X, fit=True):
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X).astype(np.float32)

    # 2. Forward pass
    def forward(self, x):
        return self.net(x)

    # 3. Training
    def fit(self, X, y, epochs=1000, verbose=200):
        Xt = torch.tensor(self.standardize(X, fit=True))
        if self.output_dim == 1:
            yt = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        else:
            yt = torch.tensor(np.asarray(y), dtype=torch.long)
        self.losses = []
        for ep in range(1, epochs + 1):
            self.train()
            self.optimizer.zero_grad()
            loss = self.criterion(self(Xt), yt)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            if verbose and ep % verbose == 0:
                print(f"Epoch {ep:4d} | loss = {loss.item():.4f}")
        return self

    # 4. Loss plot
    def plot_loss(self):
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(self.losses) + 1), self.losses)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training loss vs. epochs")
        plt.grid(alpha=.3); plt.show()

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        logits = self(torch.tensor(self.standardize(X, fit=False)))
        if self.output_dim == 1:
            return (torch.sigmoid(logits) >= 0.5).int().view(-1).numpy()
        return logits.argmax(1).numpy()

    # 5. Save to ONNX
    def save_onnx(self, path="torchnet.onnx"):
        self.eval()
        dummy = torch.randn(1, self.input_dim)
        torch.onnx.export(self, dummy, path, input_names=["input"], output_names=["logit"],
                          dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
                          opset_version=17, dynamo=False)
        print("Model saved to", path)
        return path

    # 6. Load ONNX model and predict
    def predict_onnx(self, X, path="torchnet.onnx"):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        Xs = self.standardize(X, fit=False)  # same scaler used in training
        logits = sess.run(None, {"input": Xs})[0]
        prob = 1 / (1 + np.exp(-logits))
        return (prob >= 0.5).astype(int).ravel(), prob.ravel()
