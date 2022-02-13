"""
    Данный файл содержит конечный автомат, используемый для алгоритма переноса стиля.
"""

import asyncio
import os

from aiogram import Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove
from bot_init import bot, dp
from menu import markups as mu
from menu import replies as rep


class FSMStyleTransfering(StatesGroup):
    load_content = State()
    load_style = State()
    settings_menu = State()
    confirm_run = State()

# Функция начала алгоритма переноса
async def run_style_transfer(message: types.Message):
    await FSMStyleTransfering.load_content.set()
    await message.answer(text=rep.load_content_msg, reply_markup=mu.cancel_markup, parse_mode='Markdown')

# Функция загрузки контент-изображения
async def load_content(message: types.Message, state: FSMContext):
    # Запись в словарь нужных для модели параметров
    async with state.proxy() as data:
        data['content'] = message.photo[-1].file_id  
        data['content_weight'] = str(1)
        data['n_epochs'] = str(200)
        
        # Сохранение изображения для передачи в модель
        await message.photo[-1].download(destination_file=f'photos/content/{message.photo[-1].file_id}.jpg') 

    await FSMStyleTransfering.next()
    await message.answer(text=rep.load_style_msg, reply_markup=mu.cancel_markup, parse_mode='Markdown')

# Функция загрузки стиль-изображения
async def load_style(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['style'] = message.photo[-1].file_id
        data['style_weight'] = str(1_000_000)
        await message.photo[-1].download(destination_file=f'photos/style/{message.photo[-1].file_id}.jpg')

    await FSMStyleTransfering.next()
    await message.answer(text=rep.confirm_msg, reply_markup=mu.confirm_markup, parse_mode='Markdown')


# Функция отправляет информационные сообщения при переходе в режим настройки
async def settings_menu(message: types.Message):
    await message.answer(text=rep.about_settings, parse_mode='Markdown')
    await message.answer(text=rep.settings_commands, reply_markup=mu.settings_markup, parse_mode='Markdown')


# Функция настройки параметров
async def change_setting(message: types.Message, state: FSMContext):
    try:
        param, value = message.text.split()
        float(value)
    except ValueError:
        await message.answer(text='Неизвестная команда, попробуйте еще раз', reply_markup=mu.settings_markup)  
    else:
        async with state.proxy() as data:
            if param.lower() == 'стиль':
                await message.answer(text=f'Вес стиля изменен на: {value}', reply_markup=mu.settings_markup)
                data['style_weight'] = value
            elif param.lower() == 'контент':
                await message.answer(text=f'Вес контента изменен на: {value}', reply_markup=mu.settings_markup)
                data['content_weight'] = value
            elif param.lower() == 'эпохи':
                await message.answer(text=f'Кол-во эпох изменено на: {int(value)}', reply_markup=mu.settings_markup)
                data['n_epochs'] = str(int(value))
            else:
                await message.answer(text='Неизвестная команда, попробуйте еще раз', reply_markup=mu.settings_markup)

            await message.answer(
                text='_Текущие настройки:_\n\n*style_weight* — _{}_\n*content_weight* — _{}_\n*epochs* — _{}_'.format(data['style_weight'], data['content_weight'], data['n_epochs']),
                parse_mode='Markdown'
            )

# Функция запускающая модель
async def confirm(message: types.Message, state: FSMContext):
    # Берем из словаря необходимые параметры
    async with state.proxy() as data:
        ph_id = data['content']
        content_path = f"photos/content/{ph_id}.jpg"
        content_weight = data['content_weight']

        ph_id = data['style']
        style_path = f"photos/style/{ph_id}.jpg"
        style_weight = data['style_weight']
        save_path = f"photos/saved/{ph_id}.jpg"

        n_epochs = data['n_epochs']

    await message.answer(text=rep.run_msg, reply_markup=ReplyKeyboardRemove(), parse_mode='Markdown')

    args = f'python3.8 models/style_transfer.py {content_path} {style_path} {save_path} {style_weight} {content_weight} {n_epochs}'.split()

    # Запуск подпроцесса и ожидание
    p = await asyncio.create_subprocess_exec(*args)
    await p.wait()

    await message.answer(text=rep.nst_end_msg, reply_markup=mu.startup_markup, parse_mode='Markdown')
    file = types.InputFile(save_path)
    await message.answer_photo(file) # Отправляем фото 
    await clear_files(state)
    await state.finish() # Окончание автомата


# Функция отмены алгоритма переноса
async def cancel_style_transfer(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return
    await clear_files(state)
    await state.finish()
    await message.answer(text=rep.nst_cancel_msg, reply_markup=mu.startup_markup, parse_mode='Markdown')


# Функция очистки используемых папок
async def clear_files(state: FSMContext):
    async with state.proxy() as data:
        if 'content' in data:
            content_photo_id = data['content']
            os.system(f'rm photos/content/{content_photo_id}*')
        if 'style' in data:
            style_photo_id = data['style']
            os.system(f'rm photos/style/{style_photo_id}*')
            os.system(f'rm photos/saved/{style_photo_id}*')


# Регистрация обработчиков
def register_st_handlers(dp : Dispatcher):
    dp.register_message_handler(run_style_transfer, commands=['run'], state=None)
    dp.register_message_handler(load_content, content_types=['photo'], state=FSMStyleTransfering.load_content)
    dp.register_message_handler(load_style, content_types=['photo'], state=FSMStyleTransfering.load_style)
    dp.register_message_handler(settings_menu, commands=['settings'], state=FSMStyleTransfering.settings_menu)
    dp.register_message_handler(change_setting, Text(startswith=['стиль', 'контент', 'эпохи', 'Стиль', 'Контент', 'Эпохи']), state=FSMStyleTransfering.settings_menu)
    dp.register_message_handler(confirm, commands=['confirm'], state=FSMStyleTransfering.settings_menu)
    dp.register_message_handler(confirm, commands=['confirm'], state=FSMStyleTransfering.confirm_run)
    dp.register_message_handler(cancel_style_transfer, state='*', commands=['cancel'])
