class VolatilityLSTM(nn.Module):
    """2-layer LSTM (64 units) + stock embedding (8) + dense head (64-32-1).

    The output is exp(z + c), where c is the average log RV of the training targets.
    The exponential keeps every forecast positive, which QLIKE requires.
    """

    def __init__(self, n_features: int, n_stocks: int, log_rv_mean: float,
                 hidden: int = 64, layers: int = 2, emb_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_stocks, emb_dim)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + emb_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.register_buffer("c", torch.tensor(float(log_rv_mean)))

    def forward(self, x, stock_id):
        _, (h, _) = self.lstm(x)
        z = self.head(torch.cat([h[-1], self.embedding(stock_id)], dim=1)).squeeze(-1)
        return torch.exp(z.clamp(-15, 15) + self.c)
