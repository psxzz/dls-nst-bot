from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Buttons
help_btn = KeyboardButton('/help')
about_btn = KeyboardButton('/about')
run_btn = KeyboardButton('/run')
confirm_btn = KeyboardButton('/confirm')
cancel_btn = KeyboardButton('/cancel')
settings_btn = KeyboardButton('/settings')

# Markups
startup_markup = ReplyKeyboardMarkup(resize_keyboard=True)
startup_markup.add(run_btn).add(about_btn).add(help_btn)

confirm_markup = ReplyKeyboardMarkup(resize_keyboard=True)
confirm_markup.add(confirm_btn).add(settings_btn).add(cancel_btn)

cancel_markup = ReplyKeyboardMarkup(resize_keyboard=True)
cancel_markup.add(cancel_btn)

settings_markup = ReplyKeyboardMarkup(resize_keyboard=True)
settings_markup.add(confirm_btn, cancel_btn)
