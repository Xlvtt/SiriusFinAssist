# SiriusFinAssist

## Дано
- Пользовательские запросы
- Таблицы с транзакциями
- Доступы к каким-то LLM

## Пример

![hLLMP_Serious/3.jpg](https://github.com/Xlvtt/SiriusFinAssist/blob/e3d6a9a23e312cc954055e7a6987ae9145a87ecd/LLMP_Serious/3.jpg?raw=true)

## Надо
- Собрать пайплайн который сможет на эти вопросы отвечать
- Основная сложность, что промптингом с text2sql моделью норм на эти
вопросы не ответить
- Нужно строить лэйер с ретривалом таблицы прежде чем подавать в LLM
(RAG)
- Можно попытаться решить промптингтом, можно РАГом, можно СФТ и
разные результаты получить
- (!) Главное - придумать как оценить качество решения

## Work flow
![hLLMP_Serious/5.jpg](https://github.com/Xlvtt/SiriusFinAssist/blob/6579e9d3931ae8381bf84de3320540bf1bbbd83a/LLMP_Serious/5.png)
##
![hLLMP_Serious/4.jpg](https://github.com/Xlvtt/SiriusFinAssist/blob/6579e9d3931ae8381bf84de3320540bf1bbbd83a/LLMP_Serious/4.png)


## Наша архитиктура
![hLLMP_Serious/IMG_7611.png](https://github.com/Xlvtt/SiriusFinAssist/blob/2bf9bffbef6eebab8f0feb605006210e94611c76/LLMP_Serious/IMG_7611.png)

## Архитектура базового решения
![img.png](img.png)
