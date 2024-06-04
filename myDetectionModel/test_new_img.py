import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from DetectionModel import DetectionModel


def load_model(model_path, num_classes):
    model = DetectionModel(num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)


def detect_objects(model, image_path, transform, device, class_names):
    image = preprocess_image(image_path, transform).to(device)

    with torch.no_grad():
        outputs = model(image)

    outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.shape[-1])
    bbox_preds = outputs[:, :4].cpu().numpy()
    cls_preds = torch.softmax(outputs[:, 4:], dim=1).cpu().numpy()
    cls_ids = np.argmax(cls_preds, axis=1)
    cls_scores = np.max(cls_preds, axis=1)

    img = cv2.imread(image_path)

    for bbox, cls_id, score in zip(bbox_preds, cls_ids, cls_scores):
        if score > 0.5:
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * img.shape[1])
            xmax = int(xmax * img.shape[1])
            ymin = int(ymin * img.shape[0])
            ymax = int(ymax * img.shape[0])

            class_name = class_names[cls_id]
            label = f'{class_name} {score:.2f}'

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    result_image_path = f"results/images/detection_result_{os.path.basename(image_path)}"
    cv2.imwrite(result_image_path, img)
    cv2.imshow('Detection Results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result_image_path


def main():
    # Paths and parameters
    model_path = "detection_model.pth"
    test_dataset_path = "D:/New/CompVision/dataset/images/val/"
    num_classes = 76
    class_names = [f'class_{i}' for i in range(num_classes)]  # Modify as per your dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model = load_model(model_path, num_classes).to(device)

    test_image_paths = [os.path.join(test_dataset_path, img) for img in os.listdir(test_dataset_path) if
                        img.endswith(".jpg")]
    img_path = random.choice(test_image_paths)

    detect_objects(model, img_path, transform, device, class_names)


if __name__ == "__main__":
    main()
