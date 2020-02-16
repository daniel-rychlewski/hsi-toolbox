import keras


def cao(input_shape):
    num_classes = 17
    kernel_size = 3
    lr = 0.001
    patch_size = 9
    epoch = 100
    batch_size = 100

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(300, (kernel_size), strides=(1, 1), padding='valid',
                     input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(200, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(200))
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dense(num_classes))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

    return model