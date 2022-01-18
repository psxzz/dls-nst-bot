from aiogram import types, Dispatcher

from bot_init import dp
from menu import startup_markup


async def send_welcome(message: types.Message):
    await message.answer(
        text="""Hello! I'm Style-Transfer Bot!\nAvailable commands:\n/help -- show this menu,\n/run - run style transfer""",
        reply_markup=startup_markup,
    )


def register_message_handlers(dp : Dispatcher):
    dp.register_message_handler(send_welcome, commands=['help', 'start'])