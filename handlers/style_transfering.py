import asyncio
import os

from aiogram import Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove
from bot_init import bot, dp
from menu.markups import (cancel_markup, confirm_markup, settings_markup,
                          startup_markup)
from menu.replies import (about_settings, confirm_msg, load_content_msg,
                          load_style_msg, nst_cancel_msg, nst_end_msg, run_msg, settings_commands)


class FSMStyleTransfering(StatesGroup):
    load_content = State()
    load_style = State()
    settings_menu = State()
    confirm_run = State()

async def run_style_transfer(message: types.Message):
    await FSMStyleTransfering.load_content.set()
    await message.answer(text=load_content_msg, reply_markup=cancel_markup, parse_mode='Markdown')


async def load_content(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['content'] = message.photo[-1].file_id
        data['content_weight'] = str(1)
        data['n_epochs'] = str(300)
        await message.photo[-1].download(destination_file=f'photos/content/{message.photo[-1].file_id}.jpg')

    await FSMStyleTransfering.next()
    await message.answer(text=load_style_msg, reply_markup=cancel_markup, parse_mode='Markdown')


async def load_style(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['style'] = message.photo[-1].file_id
        data['style_weight'] = str(1_000_000)
        await message.photo[-1].download(destination_file=f'photos/style/{message.photo[-1].file_id}.jpg')

    await FSMStyleTransfering.next()
    await message.answer(text=confirm_msg, reply_markup=confirm_markup, parse_mode='Markdown')


# TODO: Implement parameter settings
async def settings_menu(message: types.Message):
    await message.answer(text=about_settings, parse_mode='Markdown')
    await message.answer(text=settings_commands, reply_markup=settings_markup, parse_mode='Markdown')


async def change_setting(message: types.Message, state: FSMContext):
    try:
        param, value = message.text.split()
        int(value)
    except ValueError:
        await message.answer(text='Неизвестная команда, попробуйте еще раз', reply_markup=settings_markup)  
    else:
        async with state.proxy() as data:
            if param.lower() == 'стиль':
                await message.answer(text=f'Вес стиля изменен на: {value}', reply_markup=settings_markup)
                data['style_weight'] = value
            elif param.lower() == 'контент':
                await message.answer(text=f'Вес контента изменен на: {value}', reply_markup=settings_markup)
                data['content_weight'] = value
            elif param.lower() == 'эпохи':
                await message.answer(text=f'Кол-во эпох изменено на: {value}', reply_markup=settings_markup)
                data['n_epochs'] = value
            else:
                await message.answer(text='Неизвестная команда, попробуйте еще раз', reply_markup=settings_markup)

            await message.answer(
                text='_Текущие настройки:_\n\n*style_weight* — _{}_\n*content_weight* — _{}_\n*epochs* — _{}_'.format(data['style_weight'], data['content_weight'], data['n_epochs']),
                parse_mode='Markdown'
            )


async def confirm(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        ph_id = data['content']
        content_path = f"photos/content/{ph_id}.jpg" # 
        content_weight = data['content_weight']

        ph_id = data['style']
        style_path = f"photos/style/{ph_id}.jpg"
        style_weight = data['style_weight']
        save_path = f"photos/saved/{ph_id}.jpg"

        n_epochs = data['n_epochs']

    await message.answer(text=run_msg, reply_markup=ReplyKeyboardRemove(), parse_mode='Markdown')

    args = f'python3.8 models/style_transfer.py {content_path} {style_path} {save_path} {style_weight} {content_weight} {n_epochs}'.split()
    # Run NST subprocess
    p = await asyncio.create_subprocess_exec(*args)
    await p.wait()

    await message.answer(text=nst_end_msg, reply_markup=startup_markup, parse_mode='Markdown')
    file = types.InputFile(save_path)
    await message.answer_photo(file)
    await clear_files(state)
    await state.finish()


async def cancel_style_transfer(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return
    await clear_files(state)
    await state.finish()
    await message.answer(text=nst_cancel_msg, reply_markup=startup_markup, parse_mode='Markdown')


async def clear_files(state: FSMContext):
    async with state.proxy() as data:
        if 'content' in data:
            content_photo_id = data['content']
            os.system(f'rm photos/content/{content_photo_id}*')
        if 'style' in data:
            style_photo_id = data['style']
            os.system(f'rm photos/style/{style_photo_id}*')
            os.system(f'rm photos/saved/{style_photo_id}*')


def register_st_handlers(dp : Dispatcher):
    dp.register_message_handler(run_style_transfer, commands=['run'], state=None)
    dp.register_message_handler(load_content, content_types=['photo'], state=FSMStyleTransfering.load_content)
    dp.register_message_handler(load_style, content_types=['photo'], state=FSMStyleTransfering.load_style)
    dp.register_message_handler(settings_menu, commands=['settings'], state=FSMStyleTransfering.settings_menu)
    dp.register_message_handler(change_setting, Text(startswith=['стиль', 'контент', 'эпохи', 'Стиль', 'Контент', 'Эпохи']), state=FSMStyleTransfering.settings_menu)
    dp.register_message_handler(confirm, commands=['confirm'], state=FSMStyleTransfering.settings_menu)
    dp.register_message_handler(confirm, commands=['confirm'], state=FSMStyleTransfering.confirm_run)
    dp.register_message_handler(cancel_style_transfer, state='*', commands=['cancel'])
