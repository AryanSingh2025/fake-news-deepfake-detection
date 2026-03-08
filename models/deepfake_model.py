import tensorflow as tf

train_path = "dataset/deepfake_images/train"
test_path = "dataset/deepfake_images/test"

img_size = 128
batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=5)

test_loss, test_accuracy = model.evaluate(test_dataset)

print("Test Accuracy:", test_accuracy)

model.save("models/deepfake_model.h5")

print("Deepfake model trained and saved")