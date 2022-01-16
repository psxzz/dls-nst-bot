import logging
from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = 'BOT_TOKEN'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['help', 'start'])
async def send_welcome(message: types.Message):
    await message.reply(text="Hello! I'm dls-nst-bot!", reply=False)


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(message.text)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)