from tensorflow.keras import callbacks


def cb_earlystop(
        monitor='val_loss',
        mode='min',
        min_delta=0.001,
        patience=15,
        verbose=1
    ):

    return callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.001,
        patience=15,
        verbose=1
    )


def cb_reducelr(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        mode='min',
        min_delta=0.001,
        cooldown=0,
        min_lr=0
    ):

    return callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        mode='min',
        min_delta=0.001,
        cooldown=0,
        min_lr=0
    )


def cb_checkpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch',
        options=None
    ):

    return callbacks.ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch',
        options=None
    )


def cb_CSVLogger(
        filename,
        separator=',',
        append=False
    ):

    return callbacks.CSVLogger(
        filename,
        separator=',',
        append=False
    )