# vehicle_detection


EDA и промежуточные выводы в ноутбуке

В качестве метрики использую f1_score - метрика учитвает precision и recall

trainer.py содержит класс Trainer, который используется для обучения модели на задаче классификации транспортных средств. Этот скрипт обрабатывает весь процесс обучения, включая загрузку данных, обучение модели и валидацию.

inference.py используется для выполнения предсказаний с помощью обученной модели. Этот скрипт загружает модель, обрабатывает входные данные и выводит предсказания.

запуск предсказаний python inference.py ../videos/video_18.mp4 ../polygons.json ../output.json



