from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove
from aiogram import Dispatcher, types
from bot_init import dp, bot
from menu import confirm_markup, startup_markup, cancel_markup
import os

class StyleTransfering(StatesGroup):
    load_content = State()
    load_style = State()
    confirm_run = State()
    return_final_image = State()


async def run_style_transfer(message: types.Message):
    await StyleTransfering.load_content.set()
    await message.answer(text="Please, load the content photo", reply_markup=cancel_markup)


async def load_content(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['content'] = message.photo[-1].file_id
        await message.photo[-1].download(destination_file=f'content/{message.photo[-1].file_id}.jpg')

    await StyleTransfering.next()
    await message.answer(text="Now, load a style photo", reply_markup=cancel_markup)


async def load_style(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['style'] = message.photo[-1].file_id
        await message.photo[-1].download(destination_file=f'style/{message.photo[-1].file_id}.jpg')


    await StyleTransfering.next()
    await message.answer(text="Ok. Please confirm/cancel style transfer", reply_markup=confirm_markup)


async def confirm(message: types.Message, state: FSMContext):
        # await StyleTransfering.next()
        await message.answer(text="Ok. Running style transfering!")
        await state.finish()

async def cancel_style_transfer(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return
    
    async with state.proxy() as data:
        if 'content' in data:
            content_photo_id = data['content']
            os.system(f'rm content/{content_photo_id}*')
        if 'style' in data:
            style_photo_id = data['style']
            os.system(f'rm style/{style_photo_id}*')

    await state.finish()
    await message.answer(text='Cancelling style transfer', reply_markup=startup_markup)


def register_st_handlers(dp : Dispatcher):
    dp.register_message_handler(run_style_transfer, commands=['run'], state=None)
    dp.register_message_handler(load_content, content_types=['photo'], state=StyleTransfering.load_content)
    dp.register_message_handler(load_style, content_types=['photo'], state=StyleTransfering.load_style)
    dp.register_message_handler(confirm, commands=['confirm'], state=StyleTransfering.confirm_run)
    dp.register_message_handler(cancel_style_transfer, state='*', commands=['cancel'])