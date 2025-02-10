import numpy as np
import tensorflow as tf


np.random.seed(42)
tf.random.set_seed(42)


inches = np.linspace(1, 100, 100)
centimeters = inches * 2.54

x_train = tf.convert_to_tensor(inches.reshape(-1, 1), dtype=tf.float32)
y_train = tf.convert_to_tensor(centimeters.reshape(-1, 1), dtype=tf.float32)

W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")



def predict(x):
    return x * W + b


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

learning_rate = 1e-5
optimizer = tf.optimizers.SGD(learning_rate)

epochs = 10000

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = predict(x_train)
        loss_value = loss_fn(y_train, y_pred)

    gradients = tape.gradient(loss_value, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if epoch % 50 == 0:
        print(f"Epoch  number {epoch}'s Loss Value: {loss_value.numpy()}")

test_inches = tf.constant([10.0, 50.0, 100.0, 125, 150, 175, 200], dtype=tf.float32)
predictions = predict(tf.reshape(test_inches, (-1, 1)))

print("\nResults:")
print("Test Inches:", test_inches.numpy())
print("Expected centimeters:", test_inches.numpy() * 2.54)
print("Predicted centimeters:", predictions.numpy().flatten())