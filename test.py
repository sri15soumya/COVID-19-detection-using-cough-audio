import tensorflow as tf

model = tf.keras.models.load_model(
    "model/best_cnn_model.keras"
)

print("Loaded Successfully")
print(model.summary())