from aiogram import executor
from bot_init import dp
from handlers import messages, style_transfering
from models import style_transfer

async def on_startup(_):
    style_transfer.create_model()

messages.register_message_handlers(dp)
style_transfering.register_st_handlers(dp)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
