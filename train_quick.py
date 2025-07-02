import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Basic Config ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # QUICK TRAINING

# === Dataset Paths ===
train_dir = "dataset/train"
val_dir = "dataset/validation"

# === Data Generators ===
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# === Model Architecture ===
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base.trainable = False  # Freeze base model

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Train ===
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Save Final Model ===
model.save("deepfake_quick_model.h5")
print(" Model trained and saved as deepfake_quick_model.h5")
