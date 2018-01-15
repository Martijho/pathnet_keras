from pathnet_keras import PathNet
from dataprep import DataPrep
from analytic import Analytic
from path_search import PathSearch
from plot_pathnet import PathNetPlotter
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adagrad, Adam, RMSprop, SGD
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt


def give_model(optim):
    inp = Input([32, 32, 3])
    thread = inp

    for _ in range(1):
        thread = Conv2D(2, (3, 3), activation='relu')(thread)
        thread = Conv2D(2, (5, 5), activation='relu')(thread)
        thread = BatchNormalization()(thread)

    thread = MaxPooling2D((2, 2))(thread)
    thread = Flatten()(thread)
    thread = Dense(20, activation='relu')(thread)
    thread = Dense(2, activation='softmax')(thread)

    model = Model(inputs=inp, outputs=thread)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

optim = RMSprop()
epochs = 10
batch_size = 16

data = DataPrep()
data.cifar10()
x1, y1, x_test1, y_test1 = data.sample_dataset([0, 6])
x2, y2, x_test2, y_test2 = data.sample_dataset([2, 3])


model = give_model(optim)
model.summary()
log1 = model.fit(x1, y1, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=[x_test1, y_test1]).history['val_acc']
log2 = []
model = give_model(optim)
for e in range(epochs):
    for i in range(int(round(len(x1)/batch_size))):
        batch = np.random.randint(0, len(x1), batch_size)
        model.train_on_batch(x1[batch], y1[batch])

    pred = model.predict(x_test1)
    hits = sum(np.argmax(pred, 1) == np.argmax(y_test1, 1))
    test_acc = hits/len(x_test1)
    log2.append(test_acc)
    print(test_acc)

print(((e+1)*(i+1))/50, 'iterations/minibatches gives acc:', test_acc)

plt.plot(log1)
plt.plot(log2)
plt.legend(['fit', 'train_on_batch'])
plt.show()



