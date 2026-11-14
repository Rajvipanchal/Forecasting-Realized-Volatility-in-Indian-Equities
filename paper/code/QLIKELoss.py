class QLIKELoss(nn.Module):
    """QLIKE = RV/h - log(RV/h) - 1 (Patton, 2011). Zero only when h = RV."""

    def forward(self, h, rv):
        ratio = rv / h
        return (ratio - torch.log(ratio) - 1).mean()
