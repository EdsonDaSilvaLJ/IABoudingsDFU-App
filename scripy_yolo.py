import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# CONFIGURE AQUI OS CAMINHOS:
IMAGES_DIR = r"C:\Users\Edson\Downloads\DFUC2020_trainset_release\DFUC2020_trainset\train"              # Pasta com TODAS as imagens (.jpg)
OUTPUT_DIR = './DFUYolov5'             # Pasta que vai conter as subpastas YOLO (para YOLOv8 ou YOLOv5)
CSV_FILE = r"C:\Users\Edson\Downloads\DFUC2020_trainset_release\DFUC2020_trainset\groundtruth.csv"            # Caminho do seu CSV

# 1. Lê o CSV
df = pd.read_csv(CSV_FILE)
df['filename'] = df['filename'].astype(str) + '.jpg'

# 2. Lista todas as imagens únicas
unique_images = df['filename'].unique()

# 3. Split das imagens (70% train, 15% val, 15% test)
train_imgs, testval_imgs = train_test_split(unique_images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(testval_imgs, test_size=0.5, random_state=42)
splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

# 4. Cria as pastas (images/labels para cada split)
for split in splits:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

# 5. Função para converter bounding box para formato YOLO
def bbox_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h

# 6. Preenche imagens e labels
for split, split_imgs in splits.items():
    print(f'Processando {split}...')
    for img_name in tqdm(split_imgs):
        src_img = os.path.join(IMAGES_DIR, img_name)
        dst_img = f'{OUTPUT_DIR}/images/{split}/{img_name}'
        # Copia imagem (se ainda não copiou)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        # Lê a imagem pra pegar dimensões
        img = cv2.imread(src_img)
        h, w = img.shape[:2]
        # Filtra as boxes daquela imagem
        boxes = df[df['filename'] == img_name][['xmin', 'ymin', 'xmax', 'ymax']].values
        label_path = f'{OUTPUT_DIR}/labels/{split}/{img_name.replace(".jpg", ".txt")}'
        with open(label_path, 'w') as f:
            for box in boxes:
                x_center, y_center, box_w, box_h = bbox_to_yolo(w, h, *box)
                # Apenas uma classe (classe 0)
                f.write(f'0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n')
