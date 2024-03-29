# Отчет по проекту. Вариант 1: Перенос стиля в телеграм-ботах.


## В ходе выполнения работы:

* Реализован алгоритм переноса стиля Л. Гатиса. Качество переноса алгоритма было улучшено с помощью некоторых решений из статьи __(напр. L2 Loss заменен на L1 Loss, LBFGS заменен на Adam)__. Алгоритм перенесен в класс, имеет возможность настройки весов модели. При реализации модели проблем не возникло.

* Реализован асинхронный телеграм бот с подсказками, интерфейсом, и возможностью настройки модели из диалога. При реализации бота, возникали проблемы с асинхронной работой бота во время работы модели, решил с помощью вызова подпроцесса из **asyncio**.

* Возникли проблемы с деплоем на **Heroku** (нехватка места для модели + всего необходимого для корректной работы бота), поэтому использовал  **Google Cloud Platform**. (**Note:** на данный момент, одна фотография обрабатывается примерно 10-12 мин.) 

## Список использованных источников:

- [Пример с оф.сайта PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Статья по улучшению качества переноса](https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489)
- [Документация к aiogram](https://docs.aiogram.dev/en/latest/)
- [Asyncio subprocesses](https://docs.python.org/3/library/asyncio-subprocess.html)

### P.S.

Бота можно найти по тегу: **@neural_st_bot**

Меня можно найти по тегу: **@psxzz**