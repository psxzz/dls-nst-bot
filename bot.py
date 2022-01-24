import os

from aiogram import executor

from bot_init import dp, bot, API_TOKEN
from handlers import messages, style_transfering


WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.environ.get('PORT', 5000))
WEBHOOK_PATH = F"/webhook/{API_TOKEN}"
WEBHOOK_URL = f"https://neural-st-bot.herokuapp.com{WEBHOOK_PATH}"


async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    messages.register_message_handlers(dp)
    style_transfering.register_st_handlers(dp)


async def on_shutdown(_):
    os.system('rm photos/content/* photos/style/* photos/saved/*')


if __name__ == "__main__":
    executor.start_webhook(
        dispatcher=dp, 
        webhook_path=WEBHOOK_PATH, 
        on_startup=on_startup, 
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT
    )
    # executor.start_polling(dp, skip_updates=True, on_shutdown=on_shutdown)