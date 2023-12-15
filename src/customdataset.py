from torch.utils.data import DataLoader, Dataset
import cv2


class CustomDataset(Dataset):
    def __init__(self, data, frame_labels, transform=None):
        self.data = data
        self.frame_labels = frame_labels
        self.transform = transform

        # Формирование списка всех кадров и соответствующих меток
        self.frames = []
        self.labels = []
        for video_name, frames in self.data.items():
            self.frames.extend(frames)
            self.labels.extend(self.frame_labels[video_name])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]

        # Преобразование кадра в формат, подходящий для PyTorch
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(frame)

        return {"image": frame, "label": label}
