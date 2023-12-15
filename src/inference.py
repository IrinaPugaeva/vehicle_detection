import argparse
import torch
from torchvision import models, transforms
import torchvision
import json
import cv2

from utils import extract_polygon_frame


class Inferencer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.model = self.model.to(self.device)

    def _initialize_model(self):
        model = models.efficientnet_b1(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
        model.load_state_dict(
            torch.load("best_model.pth", map_location=torch.device("cpu"))
        )
        return model

    def inference_frame(self, frame):
        frame_tensor = frame.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(frame_tensor)
            prediction = torch.sigmoid(output) > 0.5

        return int(prediction.item())


def intervals_from_binary_sequence(seq):
    """
    Преобразует двоичную последовательность в список интервалов.
    Например, для [0, 1, 1, 1, 0] вернет [[1, 3]].
    """
    intervals = []
    start = None

    for i, value in enumerate(seq):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            intervals.append([start, i - 1])
            start = None

    # Добавляем последний интервал, если последний элемент последовательности равен 1
    if start is not None:
        intervals.append([start, len(seq) - 1])

    return intervals


def main(video_path, polygon_path, output_path):
    inferencer = Inferencer()

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

    with open(polygon_path) as file:
        polygons = json.load(file)

    video_name = video_path.split("/")[-1]

    video_outputs = []
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cropped_frame = extract_polygon_frame(frame, polygons[video_name])
        cropped_frame = data_transforms(cropped_frame)  # применяем преобразования
        output = inferencer.inference_frame(cropped_frame)
        video_outputs.append(output)

    video_intervals = intervals_from_binary_sequence(video_outputs)
    json.dump({video_name: video_intervals}, open(output_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("polygons_path", help="Path to the polygons file")
    parser.add_argument("output_path", help="Path for the output file")
    
    args = parser.parse_args()
    
    main(args.video_path, args.polygons_path, args.output_path)
