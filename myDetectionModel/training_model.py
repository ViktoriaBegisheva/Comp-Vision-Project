import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from CustomDataset import CustomDataset
from DetectionModel import DetectionModel

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(labels)

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    batch_size = 16

    path_img_train = 'D:/New/CompVision/dataset/images/train'
    path_img_val = 'D:/New/CompVision/dataset/images/val'
    path_labl_train = 'D:/New/CompVision/dataset/labels/train'
    path_labl_val = 'D:/New/CompVision/dataset/labels/val'

    train_dataset = CustomDataset(img_dir=path_img_train, label_dir=path_labl_train, transform=transform)
    val_dataset = CustomDataset(img_dir=path_img_val, label_dir=path_labl_val, transform=transform)
    print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print(len(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 76
    model = DetectionModel(num_classes).to(device)

    criterion_bbox = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_val_loss = float('inf')
    model_save_path = "detection_model_new.pth"

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc='Training', unit='batch')
        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            outputs = outputs.view(-1, outputs.shape[-1])

            bbox_preds = outputs[:, :4]
            cls_preds = outputs[:, 4:]

            bbox_targets = []
            cls_targets = []
            for label in labels:
                bbox_targets.append(label[:, :4])
                cls_targets.append(label[:, 4].long())
            bbox_targets = torch.cat(bbox_targets, dim=0)
            cls_targets = torch.cat(cls_targets, dim=0)

            if bbox_preds.shape != bbox_targets.shape:
                min_size = min(bbox_preds.shape[0], bbox_targets.shape[0])
                bbox_preds = bbox_preds[:min_size]
                bbox_targets = bbox_targets[:min_size]

            if cls_preds.shape[0] != cls_targets.shape[0]:
                min_size = min(cls_preds.shape[0], cls_targets.shape[0])
                cls_preds = cls_preds[:min_size]
                cls_targets = cls_targets[:min_size]

            loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
            loss_cls = criterion_cls(cls_preds, cls_targets)

            loss = loss_bbox + loss_cls
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix({'loss': running_loss / ((train_loader_tqdm.n + 1) * batch_size)})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc='Validating', unit='batch')
        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                images = images.to(device)
                labels = [label.to(device) for label in labels]

                outputs = model(images)
                outputs = outputs.permute(0, 2, 3, 1).contiguous()
                outputs = outputs.view(-1, outputs.shape[-1])

                bbox_preds = outputs[:, :4]
                cls_preds = outputs[:, 4:]

                bbox_targets = []
                cls_targets = []
                for label in labels:
                    bbox_targets.append(label[:, :4])
                    cls_targets.append(label[:, 4].long())
                bbox_targets = torch.cat(bbox_targets, dim=0)
                cls_targets = torch.cat(cls_targets, dim=0)

                if bbox_preds.shape != bbox_targets.shape:
                    min_size = min(bbox_preds.shape[0], bbox_targets.shape[0])
                    bbox_preds = bbox_preds[:min_size]
                    bbox_targets = bbox_targets[:min_size]

                if cls_preds.shape[0] != cls_targets.shape[0]:
                    min_size = min(cls_preds.shape[0], cls_targets.shape[0])
                    cls_preds = cls_preds[:min_size]
                    cls_targets = cls_targets[:min_size]

                loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
                loss_cls = criterion_cls(cls_preds, cls_targets)

                loss = loss_bbox + loss_cls
                val_loss += loss.item() * images.size(0)
                val_loader_tqdm.set_postfix({'val_loss': val_loss / ((val_loader_tqdm.n + 1) * batch_size)})

        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_save_path)
            print(f"Model saved at epoch {epoch+1}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
