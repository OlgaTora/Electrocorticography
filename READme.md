### Для запуска проекта:

1. Установите docker (есть версия с интерфейсом)
```
https://docs.docker.com/
```
2. Клонируйте этот репозитрий
```
https://github.com/OlgaTora/Electrocorticography.git
```
3.Перейдите в папку проекта
```
cd Electrocorticography/app
```
4.При первом запуске выполнить в терминале следующие команды:
```
docker build . -t rats
docker run -it --name rats rats
```
docker сам создаст виртуальное окружение 
и установит все необходимые программы внутри контейнера.

При повторных использованиях:
```
docker start -i rats
```
5. Запустите в терминале приложение согласно вашей ОС\
Windows
```
python main.py
```
Linux/MacOS
```
python3 main.py
```
6. Перейдите по адресу: 
```
http://127.0.0.1:8000/
```
