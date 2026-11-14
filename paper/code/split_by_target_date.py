def split_by_target_date(p: pd.DataFrame):
    """Assign each row to a split by the date of its TARGET (t+1).

    This prevents the last training row from using a validation-day outcome.
    """
    d = p["Date_t1"]
    train = p[(d >= C.TRAIN_START) & (d <= C.TRAIN_END)].copy()
    valid = p[(d >= C.VALID_START) & (d <= C.VALID_END)].copy()
    test = p[(d >= C.TEST_START) & (d <= C.TEST_END)].copy()
    return train, valid, test
