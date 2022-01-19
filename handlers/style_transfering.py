from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove
from aiogram import Dispatcher, types
from bot_init import dp, bot
from menu import confirm_markup, startup_markup, cancel_markup
from models import style_transfer
import os


class FSMStyleTransfering(StatesGroup):
    load_content = State()
    load_style = State()
    confirm_run = State()
    return_image = State()

async def run_style_transfer(message: types.Message):
    await FSMStyleTransfering.load_content.set()
    await message.answer(text="Please, load the content photo", reply_markup=cancel_markup)


async def load_content(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['content'] = message.photo[-1].file_id
        await message.photo[-1].download(destination_file=f'photos/content/{message.photo[-1].file_id}.jpg')

    await FSMStyleTransfering.next()
    await message.answer(text="Now, load a style photo", reply_markup=cancel_markup)


async def load_style(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['style'] = message.photo[-1].file_id
        await message.photo[-1].download(destination_file=f'photos/style/{message.photo[-1].file_id}.jpg')

    await FSMStyleTransfering.next()
    await message.answer(text="Ok. Please confirm/cancel style transfer", reply_markup=confirm_markup)


async def confirm(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        ph_id = data['content']
        content_path = f"photos/content/{ph_id}.jpg" # 
        ph_id = data['style']
        style_path = f"photos/style/{ph_id}.jpg"
        save_path = f"photos/saved/{ph_id}.jpg"

    await message.answer(text="Ok. Running style transfering!", reply_markup=ReplyKeyboardRemove())

    # TODO: Fix bot idling 
    await style_transfer.transform(content_path, style_path, save_path)

    await message.answer(text='Your photo is ready!', reply_markup=startup_markup)
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
    await message.answer(text='Cancelling style transfer', reply_markup=startup_markup)


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
    dp.register_message_handler(confirm, commands=['confirm'], state=FSMStyleTransfering.confirm_run)
    dp.register_message_handler(cancel_style_transfer, state='*', commands=['cancel'])