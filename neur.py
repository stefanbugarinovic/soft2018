from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
  
#Obucavanje vestacke neuronske mreze
def train_and_test_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data() #ucitavanje iz mnist dataset

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')    #tranin datset sa 60 hiljada brojeva
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')       #test dataset sa 10 hiljada brojeva

    x_train = x_train / 255 #elementi matrice su vrednosti 0 ili 255
    x_test = x_test / 255   

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

#kreiranje modela neuronske mreze
def create_model(x_train, y_train, epochs=1):
    batch_size = 32
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1)

    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs, validation_data=(x_train, y_train),
        steps_per_epoch=len(x_train) / batch_size)
    model.save('model.h5')
    return model, datagen
