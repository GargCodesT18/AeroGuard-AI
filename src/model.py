import torch
import torch.nn as nn


class LSTMRULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


@torch.no_grad()
def predict_with_uncertainty(model, x, n_samples=100):
    if n_samples < 2:
        raise ValueError("Monte Carlo dropout requires at least two draws")
    was_training = model.training
    model.train()
    try:
        predictions = torch.stack([model(x) for _ in range(n_samples)])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0, correction=0)
        return mean.cpu().numpy(), std.cpu().numpy()
    finally:
        model.train(was_training)
