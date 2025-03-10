from tensorflow.keras import layers, models, losses, metrics, callbacks

class UNet:
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(self.input_shape)

        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        c1 = layers.Dropout(0.1)(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        c2 = layers.Dropout(0.1)(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        c3 = layers.Dropout(0.2)(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        c4 = layers.Dropout(0.2)(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        c5 = layers.Dropout(0.3)(c5)

        # Decoder
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        c6 = layers.Dropout(0.2)(c6)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        c7 = layers.Dropout(0.2)(c7)

        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        c8 = layers.Dropout(0.1)(c8)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        c9 = layers.Dropout(0.1)(c9)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return models.Model(inputs, outputs)

    def compile(self, optimize, loss):
        self.model.compile(optimizer=optimize, loss=loss, metrics=[metrics.MeanAbsoluteError()])

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=32):
        log_dir = "logs/fit"
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = callbacks.ModelCheckpoint("unet_best_model.keras", save_best_only=True)
        early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback]
            )

    def predict(self, X_test):
        return self.model.predict(X_test)