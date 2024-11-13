from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from facenet_pytorch import MTCNN
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Класс для предобработки изображений
class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)

    def fit(self, X, y=None):
        return self  # Возвращаем сам объект для совместимости с Pipeline

    def transform(self, image_path):
        # Загрузка изображения
        image = Image.open(image_path)

        # Приведение к стандартному размеру
        standard_size = (1024, 1024)
        resized_image = image.resize(standard_size)

        # Обнаружение лиц
        boxes, _ = self.mtcnn.detect(image)

        if boxes is not None and len(boxes) == 1:
            print("На изображении одно лицо")
        else:
            print("Либо нет лица, либо несколько лиц")

        # Обнаружение лиц и ключевых точек
        boxes, probs, landmarks = self.mtcnn.detect(resized_image, landmarks=True)

        if landmarks is not None:
            landmark = landmarks[0]  # Используем первое лицо
            
            # Закомментированные строки, связанные с областью вокруг глаз
            left_eye, right_eye = landmark[0], landmark[1]
            eye_region = resized_image.crop((left_eye[0] - 70, left_eye[1] - 70, right_eye[0] + 70, right_eye[1] + 70))
            eye_region = eye_region.resize((224, 224)).convert('L')
            #eye_region.show()
            eye_region_array = np.array(eye_region)
            eye_region_array = np.stack([eye_region] * 3, axis=-1)  # Дублируем канал
            #eye_region_array = np.array(eye_region) / 255.0  # Нормализация

            mouth_left, mouth_right = landmark[3], landmark[4]

            # Выделение и преобразование области вокруг рта
            mouth_region = resized_image.crop((mouth_left[0] - 70, mouth_left[1] - 70, mouth_right[0] + 70, mouth_right[1] + 70))
            mouth_region = mouth_region.resize((224, 224)).convert('L')
            
            mouth_region_array = np.array(mouth_region)
            mouth_region_array = np.stack([mouth_region_array] * 3, axis=-1)  # Дублируем канал
            # Конвертация в массив numpy и нормализация
            #mouth_region_array = np.array(mouth_region) / 255.0

            return eye_region_array, mouth_region_array
        else:
            raise ValueError("Не удалось обнаружить ключевые точки на изображении.")

def load_model(weights_path, dense_units, dropout_rate):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # Замораживаем базовую модель

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),  # Можно изменить по необходимости
        layers.Dense(1, activation='sigmoid')  # Бинарная классификация
    ])

    # Загрузка весов, если они существуют
    try:
        model.load_weights(weights_path)
        print(f"Весы для {weights_path} загружены успешно")
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")

    return model

# Загрузка моделей для глаз и губ
model_lips = load_model("best_model_weights_lips.h5", 128, 0)
model_eyes = load_model("best_model_weights_eyes.h5", 256, 0.2)

# Функции для предсказания по глазам и губам
def predict_eyes(eye_region):
    processed_image = np.expand_dims(eye_region, axis=0)
    return model_eyes.predict(processed_image)[0][0]

def predict_lips(mouth_region):
    processed_image = np.expand_dims(mouth_region, axis=0)
    return model_lips.predict(processed_image)[0][0]