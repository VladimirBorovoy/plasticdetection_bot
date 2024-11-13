import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import numpy as np
from config import TOKEN
from image_processing import ImagePreprocessor, load_model, predict_lips, predict_eyes
import asyncio

# Загружаем модель для губ один раз при старте
model_lips = load_model(dense_units=128, dropout_rate=0.0, weights_path="best_model_weights_lips.h5")

# Загружаем модель для глаз
model_eyes = load_model(dense_units=256, dropout_rate=0.2, weights_path="best_model_weights_eyes.h5")

# Функция обработки команд "start"
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Отправь мне фото, и я определю вероятность наличия пластической операции на губах и глазах.")

# Обработка фото
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file = await update.message.photo[-1].get_file()
    file_path = "user_photo.jpg"
    await file.download_to_drive(file_path)

    try:
        preprocessor = ImagePreprocessor()

        # Выполняем предобработку и предсказание в фоновом потоке
        loop = asyncio.get_running_loop()
        eye_region, mouth_region = await loop.run_in_executor(None, preprocessor.transform, file_path)

        # Предсказания для глаз и губ
        probability_eyes = await loop.run_in_executor(None, predict_eyes, eye_region)
        probability_lips = await loop.run_in_executor(None, predict_lips, mouth_region)

        # Отправляем результаты пользователю
        await update.message.reply_text(f"Вероятность пластической операции на глазах: {probability_eyes:.4f}")
        await update.message.reply_text(f"Вероятность пластической операции на губах: {probability_lips:.4f}")

    except ValueError as e:
        if "Либо нет лица, либо несколько лиц" in str(e):
            await update.message.reply_text("На изображении должно быть одно лицо. Попробуй отправить другое фото.")
        elif "Не удалось обнаружить ключевые точки на изображении." in str(e):
            await update.message.reply_text("Не удалось обнаружить глаза и губы на изображении. Попробуй отправить другое фото.")
    finally:
        os.remove(file_path)

# Запуск бота
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()

if __name__ == "__main__":
    main()