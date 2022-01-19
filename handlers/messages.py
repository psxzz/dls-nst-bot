from aiogram import types, Dispatcher

from bot_init import dp
from menu.markups import startup_markup
from menu.replies import welcome_msg, about_nst_msg


async def send_welcome(message: types.Message):
    await message.answer(
        text=welcome_msg,
        reply_markup=startup_markup,
    )

async def about_nst(message: types.Message):
    await message.answer(
        text=about_nst_msg,
        reply_markup=startup_markup
    )


def register_message_handlers(dp : Dispatcher):
    dp.register_message_handler(send_welcome, commands=['help', 'start'])
    dp.register_message_handler(about_nst, commands=['about'])