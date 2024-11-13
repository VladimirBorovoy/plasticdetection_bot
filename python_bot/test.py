from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me. I am Plastic Detection bot!')


app = ApplicationBuilder().token("8154917555:AAEQK4-7vbS4UkeLyWjMjVcrPR0zez28XAI").build()

app.add_handler(CommandHandler("start", start_command))

app.run_polling()