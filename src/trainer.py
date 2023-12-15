import copy
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm
import yaml

from customdataset import CustomDataset
from utils import get_frame, get_frame_count, extract_polygon_frame, get_frame_labels


class Trainer:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

        self.model = self._initialize_model(self.config["model"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config["step_size"],
            gamma=self.config["gamma"],
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _initialize_model(self, model_name):
        if model_name == "efficientnet_b1":
            model = models.efficientnet_b1(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 1)
        return model

    def preprocess(self, polygons, time_intervals):
        IMAGE_WIDTH = 240  # стандартный размер входного изображения для EfficientNet-B1 составляет 240x240 пикселей
        IMAGE_HIGHT = 480

        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # transformations
        data_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_WIDTH, IMAGE_HIGHT)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # Загрузка и обработка данных
        data = {}
        for video_name, polygon in polygons.items():
            cropped_frames = []
            video_path = f"../videos/{video_name}"
            video = cv2.VideoCapture(video_path)
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                cropped_frame = extract_polygon_frame(frame, polygon)
                cropped_frames.append(cropped_frame)
            data[video_name] = cropped_frames

        frame_labels = {}
        for video_name, intervals in time_intervals.items():
            video_path = f"../videos/{video_name}"
            labels = get_frame_labels(video_path, intervals)
            frame_labels[video_name] = labels

        # Разделение данных на тренировочные и тестовые
        train_frames = {}
        test_frames = {}

        # Перебираем ключи и разделяем данные на тренировочный и тестовый наборы
        for i, (video_name, frames) in enumerate(data.items()):
            if i in [1, 3, 8, 17]:
                test_frames[video_name] = frames
            elif i not in [0, 1, 3, 8, 17, 4]:
                train_frames[video_name] = frames

        train_labels = {}
        test_labels = {}

        # Перебираем ключи и разделяем метки на тренировочный и тестовый наборы
        video_names = list(frame_labels.keys())
        for i, video_name in enumerate(video_names):
            if i in [1, 3, 8, 17]:
                test_labels[video_name] = frame_labels[video_name]
            elif i not in [0, 1, 3, 8, 17, 4]:
                train_labels[video_name] = frame_labels[video_name]

        # Создание экземпляров CustomDataset
        self.train_dataset = CustomDataset(train_frames, train_labels, data_transforms)
        self.test_dataset = CustomDataset(test_frames, test_labels, data_transforms)

        # Создание DataLoader'ов
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)

    def train(self, n_epochs):
        best_val_f1 = float("-inf")
        best_model_wts = copy.deepcopy(self.model.state_dict())

        n_epochs = 1
        for epoch in range(n_epochs):
            self.model.train()  # Переключаем модель в режим обучения
            epoch_loss = 0
            all_labels = []
            all_predictions = []

            for data in tqdm(self.train_loader):
                x_batch, y_batch = data["image"], data["label"]
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.unsqueeze(1).float()
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                predicted = (
                    torch.sigmoid(outputs).data > 0.5
                )  # Пороговое значение для классификации
                all_labels.extend(y_batch.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            self.scheduler.step()
            train_f1 = f1_score(all_labels, all_predictions)
            print(
                f"Epoch: {epoch}, Train Loss: {epoch_loss / len(self.train_loader)}, Train F1: {train_f1}"
            )

            # Валидация
            self.model.eval()  # Переключаем модель в режим валидации
            val_loss = 0
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for data in self.test_loader:
                    x_batch, y_batch = data["image"], data["label"]
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.unsqueeze(1).float()
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(x_batch)
                    val_loss += self.criterion(outputs, y_batch).item()
                    predicted = torch.sigmoid(outputs).data > 0.5
                    all_labels.extend(y_batch.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            val_f1 = f1_score(all_labels, all_predictions)
            print(
                f"Epoch: {epoch}, Val Loss: {val_loss / len(self.test_loader)}, Val F1: {val_f1}"
            )

            # Проверка и сохранение наилучшей модели на основе F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), "best_model.pth")

        # Загрузка лучших весов модели после завершения тренировки
        self.model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    with open("../polygons.json") as file:
        polygons = json.load(file)

    with open("../time_intervals.json") as file:
        time_intervals = json.load(file)

    trainer = Trainer("config.yaml")
    trainer.preprocess(polygons, time_intervals)
    trainer.train(n_epochs=1)
