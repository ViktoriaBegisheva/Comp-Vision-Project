import os
import torch
import cv2

# Путь к вашей кастомной модели
model_path = "yolov5/runs/train/exp23/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Путь к видео
video_path = "videos/00067cfb-caba8a02.mp4"
output_video_path = f"results/videos/output_video - {video_path.split('/')[-1]}.mp4"

# Загрузка видео
cap = cv2.VideoCapture(video_path)
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Получение информации о видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Настройка VideoWriter для записи выходного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    print("start")
    ret, frame = cap.read()
    if not ret:
        break

    # Применение модели к кадру
    results = model(frame)
    df = results.pandas().xyxy[0]
    if [i for i in df.iterrows()] != []: print("find sign")

    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax, confidence, class_id, name = row

        # Рисование прямоугольников и меток
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, f'{name} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Запись кадра в выходное видео
    out.write(frame)
    frame_count += 1
    print(f"Processed frame {frame_count}")
print("fin")
# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
