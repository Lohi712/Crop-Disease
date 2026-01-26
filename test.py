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
model.save(MODEL_NAME)
print("✅ Final model saved")
label_map = {}
with open(LABEL_MAP, "r") as f:
    for line in f:
        idx, label = line.strip().split(":")
        label_map[int(idx)] = label
model = tf.keras.models.load_model(MODEL_NAME)

dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
model(dummy_input)  # 👈 IMPORTANT: call model directly, not predict()

def is_blurry(img_path):
    img = cv2.imread(img_path)
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    threshold = 100.0  # You can adjust this threshold
    return variance < threshold
def generate_gradcam(model, img_path):
    # Ensure the model is built/called before accessing model.input/model.output
    # (Sequential models don't have symbolic inputs/outputs until called once)

    # 🔥 Always use last conv layer from MobileNetV2
    base_model = model.layers[0]
    last_conv_layer = None

    for layer in reversed(base_model.layers):
        if "conv" in layer.name.lower():
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in base model.")

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 Build the graph (required so model.inputs/model.outputs exist)
    _ = model(img_array, training=False)

    # 🔥 Build functional Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)

    original = cv2.imread(img_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image from path: {img_path}")

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    output_path = "heatmap_output.jpg"
    cv2.imwrite(output_path, overlay)
    return output_path
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
    disease_name = label_map[top_idx]

    # Epic 3: Unknown/Unclear Result Story
    if confidence < confidence_threshold:
        return {
            "status": "unknown",
            "disease": "Unknown/Unclear",
            "message": "Please consult a human expert or retake the photo.",
            "confidence": round(confidence * 100, 2)
        }
    
    # Epic 3: Healthy Plant Story
    # Check if the predicted label contains the word "healthy"
    is_healthy = "healthy" in disease_name.lower()

    return {
        "status": "success",
        "disease": disease_name,
        "is_healthy": is_healthy,
        "confidence": round(confidence * 100, 2)
    }
print("Final Training Accuracy:", history.history["accuracy"][-1])
print("Final Validation Accuracy:", history.history["val_accuracy"][-1])
def predict_top_k(img_path, k=3, min_conf=0.05):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        conf = float(preds[idx])
        if conf >= min_conf:
            results.append({
                "disease": label_map[idx],
                "confidence": round(conf * 100, 2)
            })

    return results
def estimate_severity_from_heatmap(heatmap_path):
    heatmap = cv2.imread(heatmap_path)
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate "hot" infected zones
    _, binary_map = cv2.threshold(heatmap_gray, 200, 255, cv2.THRESH_BINARY)

    infected_area = np.sum(binary_map == 255)
    total_area = binary_map.shape[0] * binary_map.shape[1]

    ratio = infected_area / total_area

    if ratio < 0.10:
        severity = "Low"
    elif ratio < 0.40:
        severity = "Medium"
    else:
        severity = "High"

    return {
        "severity": severity,
        "infected_ratio": round(ratio * 100, 2)
    }
def full_predict_with_heatmap(img_path):
    result = predict_disease(img_path)

    if result["status"] != "success":
        return result

    heatmap_path = generate_gradcam(model, img_path)
    severity_info = estimate_severity_from_heatmap(heatmap_path)

    result["heatmap"] = heatmap_path
    result["severity"] = severity_info["severity"]
    result["infected_ratio"] = severity_info["infected_ratio"]

    return result
if __name__ == "__main__":
    TEST_IMAGE = r'0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG'

    result = full_predict_with_heatmap(TEST_IMAGE)
    print("\n--- Final Prediction with Severity ---")
    print(result)
