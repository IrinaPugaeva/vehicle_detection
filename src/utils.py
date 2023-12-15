import cv2
import numpy as np


def get_frame(video_path, frame_number):
    video = cv2.VideoCapture(video_path)
    # Перемещаемся к указанному кадру
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()
    return frame


# Функция для подсчета кол-ва кадров в видео
def get_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    # Переменная для подсчета кадров
    frame_count = 0

    # Чтение каждого кадра
    while True:
        ret, frame = video.read()
        # Если кадр не был успешно считан (конец файла), выходим из цикла
        if not ret:
            break
        frame_count += 1

    video.release()

    return frame_count


# Функция принимает кадр и координаты многоугольника и возвращает вырезанный полигон
def extract_polygon_frame(frame, polygon):
    # Создаем маску с теми же размерами, что и кадр
    mask = np.zeros_like(frame)

    # Заполняем маску
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], (255, 255, 255))

    # Применяем маску к кадру
    masked_frame = cv2.bitwise_and(frame, mask)

    # Находим ограничивающий прямоугольник для полигона
    x, y, w, h = cv2.boundingRect(np.array(polygon, np.int32))

    # Вырезаем полигон из кадра
    cropped_frame = masked_frame[y : y + h, x : x + w]

    return cropped_frame


def get_frame_labels(video_path, intervals):
    frame_count = get_frame_count(video_path)

    # Инициализация массива меток нулями
    labels = [0] * frame_count

    # Установка метки 1 для кадров в интервалах
    for interval in intervals:
        for i in range(interval[0], interval[1] + 1):
            labels[i] = 1

    return labels
