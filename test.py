import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import cv2

BASE_DIR = Path.cwd()
DATASET_PATH = BASE_DIR / "dataset_filtered"
LABEL_MAP = BASE_DIR / "label_map.txt"
MODEL_NAME = BASE_DIR / "crop_disease_model.keras"
EPOCHS = 12

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found at {DATASET_PATH}\n"
        "Please download the dataset and place it in the project root as 'dataset_filtered/'"
    )

train_datagen = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
NUM_CLASSES = train_data.num_classes
labels = train_data.class_indices
with open(LABEL_MAP, "w") as f:
    for label, idx in labels.items():
        f.write(f"{idx}:{label}\n")

print("✅ Label map saved to", LABEL_MAP)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False 
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)
]
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)
print("Final Training Accuracy:", history.history["accuracy"][-1])
label_map = {}
with open(LABEL_MAP, "r") as f:
    for line in f:
        idx, label = line.strip().split(":")
        label_map[int(idx)] = label
def is_blurry(img_path):
    img = cv2.imread(img_path)
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    threshold = 100.0  # You can adjust this threshold
    return variance < threshold

def generate_gradcam(model, img_path, layer_name=None):
    # If no layer_name is provided, automatically find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, 'activation') and 'conv' in layer.name:
                layer_name = layer.name
                break
        else:
            # If not found in top-level, try inside submodels (e.g., MobileNetV2 as base)
            for layer in reversed(model.get_layer('mobilenetv2_1.00_224').layers):
                if hasattr(layer, 'activation') and 'conv' in layer.name:
                    layer_name = 'mobilenetv2_1.00_224/' + layer.name
                    break
    if layer_name is None:
        raise ValueError("No convolutional layer found in the model.")

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # If using submodel, get the correct layer
    if '/' in layer_name:
        base_name, sublayer = layer_name.split('/')
        conv_layer = model.get_layer(base_name).get_layer(sublayer).output
        grad_model = tf.keras.models.Model([model.inputs], [conv_layer, model.output])
    else:
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)

    original = cv2.imread(img_path)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("heatmap_output.jpg", overlay)
    return "heatmap_output.jpg"

def predict_disease(img_path, confidence_threshold=0.45):
    if is_blurry(img_path):
        return {
            "status": "error",
            "message": "Image is too blurry. Please retake photo.",
            "confidence": 0.0
        }
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    confidence = float(preds[top_idx])

    if confidence < confidence_threshold:
        return {
            "status": "unknown",
            "disease": "Unknown",
            "confidence": round(confidence * 100, 2)
        }
    return {
        "status": "success",
        "disease": label_map[top_idx],
        "confidence": round(confidence * 100, 2)
    }
print("Final Training Accuracy:", history.history["accuracy"][-1])
print("Final Validation Accuracy:", history.history["val_accuracy"][-1])

def full_predict_with_heatmap(img_path):
    result = predict_disease(img_path)

    if result["status"] != "success":
        return result

    heatmap_path = generate_gradcam(model, img_path)

    result["heatmap"] = heatmap_path
    return result
result =predict_disease("0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG")
print(result)

def predict_top_k(img_path, k=3):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "disease": label_map[idx],
            "confidence": round(float(preds[idx]) * 100, 2)
        })

    return results

if __name__ == "__main__":
    result = full_predict_with_heatmap("0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG")
    print(result)

