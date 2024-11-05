import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

## 1. MNIST data import
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ## 2. preprocessing
# x_train, x_test = x_train/255.0, x_test/255.0

# ## 3. model define
# model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28,28)),
#         tf.keras.layers.Dense(512, activation=tf.nn.relu),
#         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# ## 4. model compile
# model.compile(optimizer='adam',
#                             loss = 'sparse_categorical_crossentropy',
#                             metrics=['accuracy'])

# ## 5. training model
# model.fit(x_train, y_train, epochs=5)

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('acc: ', test_acc)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(train_imgaes[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# print(train_imgaes.shape)
# print(len(train_labels))
# print(train_labels)

# plt.figure()
# plt.imshow(train_imgaes[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('accuracy: ', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print("print: ", predictions[0])

print(np.argmax(predictions[0]))