from aiogram import executor
from bot_init import dp
from handlers import messages, style_transfering

messages.register_message_handlers(dp)
style_transfering.register_st_handlers(dp)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
