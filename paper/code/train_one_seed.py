def train_one_seed(X, meta, train_idx, valid_idx, n_stocks, seed, verbose=True):
    """Train with early stopping on validation QLIKE; return model and history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    c = float(np.log(meta[C.TARGET].to_numpy()[train_idx]).mean())
    model = VolatilityLSTM(X.shape[2], n_stocks, c)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-6)
    loss_fn = QLIKELoss()
    train_dl = _loader(X, meta, train_idx, True)
    valid_dl = _loader(X, meta, valid_idx, False)

    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        tr = 0.0
        for xb, yb, sb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(yb)
        tr /= len(train_idx)
        model.eval()
        with torch.no_grad():
            va = sum(loss_fn(model(xb, sb), yb).item() * len(yb) for xb, yb, sb in valid_dl)
        va /= len(valid_idx)
        sched.step(va)
        hist.append({"Seed": seed, "Epoch": epoch, "Train_QLIKE": tr, "Validation_QLIKE": va,
                     "Learning_Rate": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"seed {seed} | epoch {epoch:2d} | train {tr:.4f} | valid {va:.4f}", flush=True)
        if va < best - 1e-5:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)
