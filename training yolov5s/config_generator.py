import os
from glob import glob
import yaml
from sklearn.model_selection import train_test_split

# Створення необхідних директорій
base_dir = 'D:/New/CompVision/dataset'
os.makedirs(f'{base_dir}/images/train', exist_ok=True)
os.makedirs(f'{base_dir}/images/val', exist_ok=True)
os.makedirs(f'{base_dir}/labels/train', exist_ok=True)
os.makedirs(f'{base_dir}/labels/val', exist_ok=True)

# Завантаження всіх зображень
img_list = glob(f'{base_dir}/images_all/*.jpg')  # Замініть на ваш шлях до зображень
print(f'Загальна кількість зображень: {len(img_list)}')

# Розподіл на тренувальний та валідаційний набори
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)

# Переміщення файлів до відповідних директорій
def move_files(file_list, destination):
  for file_path in file_list:
    file_name = os.path.basename(file_path)
    os.rename(file_path, f'{destination}/{file_name}')

move_files(train_img_list, f'{base_dir}/images/train')
move_files(val_img_list, f'{base_dir}/images/val')

# Пошук та переміщення відповідних анотацій
def move_annotations(file_list, destination):
  for file_path in file_list:
    file_name = os.path.basename(file_path).replace('.jpg', '.txt')
    annotation_path = f'{base_dir}/txts (YOLO)/{file_name}'
    if os.path.exists(annotation_path):
      os.rename(annotation_path, f'{destination}/{file_name}')
    else:
      print(f'Анотація для {file_name} не знайдена')

move_annotations(train_img_list, f'{base_dir}/labels/train')
move_annotations(val_img_list, f'{base_dir}/labels/val')

# Створення файлів train.txt та val.txt
train_img_paths = [f'{base_dir}/images/train/{os.path.basename(path)}' for path in train_img_list]
val_img_paths = [f'{base_dir}/images/val/{os.path.basename(path)}' for path in val_img_list]

with open(f'{base_dir}/train.txt', 'w') as f:
  f.write('\n'.join(train_img_paths) + '\n')

with open(f'{base_dir}/val.txt', 'w') as f:
  f.write('\n'.join(val_img_paths) + '\n')

# Створення dataset.yaml
data = {
  'train': f'{base_dir}/train.txt',
  'val': f'{base_dir}/val.txt',
  'nc': 76,  # Кількість класів
  'names': [f'class{i}' for i in range(1, 77)]  # Імена класів
}

with open(f'{base_dir}/dataset.yaml', 'w') as f:
  yaml.dump(data, f)

print("Підготовка даних завершена")
