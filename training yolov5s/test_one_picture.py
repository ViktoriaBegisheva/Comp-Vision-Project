import os
import random

import torch
import cv2

model_path = "yolov5/runs/train/exp23/weights/best.pt"  # Вкажіть шлях до вашої кастомної моделі
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

test_dataset_path = "D:/New/CompVision/dataset/images/val/"
test_image_paths = [os.path.join(test_dataset_path, img) for img in os.listdir(test_dataset_path) if img.endswith(".jpg")]

i=0
while(i<50):
    img_path = random.choice(test_image_paths)
    img = cv2.imread(img_path)

    results = model(img)
    df = results.pandas().xyxy[0]
    if [i for i in df.iterrows()] == []: continue
    else:
        i+=1

    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax, confidence, class_id, name = row

        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(img, f'{name} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(f"results/images/detection_result{img_path.split('/')[-1]}.jpg", img)
        print("name:", img_path.split('/')[-1])

# Відображення зображення з детекціями
# cv2.imshow('Detection Results', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
