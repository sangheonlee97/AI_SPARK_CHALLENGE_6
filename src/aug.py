import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

# 이미지 및 마스크를 포함하는 디렉토리 경로 설정
data_dir = "C:/Study/AI_SPARK_CHALLENGE_6/resource/dataset/"
image_dir = os.path.join(data_dir, "train_img")
mask_dir = os.path.join(data_dir, "train_mask")
a_image_dir = os.path.join(data_dir, "aug_img")
a_mask_dir = os.path.join(data_dir, "aug_mask")
# ImageDataGenerator 초기화 및 증강 파라미터 설정
data_gen_args = dict(
                     horizontal_flip=True,
                     vertical_flip=True,
                    )
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed = 0
# 이미지 및 마스크 파일들에 대한 증강 진행
for img_file, mask_file in zip(os.listdir(image_dir), os.listdir(mask_dir)):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    # 이미지와 마스크를 함께 증강
    seed += 1
    img_gen = image_datagen.flow(img.reshape(1, *img.shape), batch_size=1, seed=seed)
    mask_gen = mask_datagen.flow(mask.reshape(1, *mask.shape), batch_size=1, seed=seed)

    augmented_img = img_gen.next()[0]
    augmented_mask = mask_gen.next()[0]

    # 생성된 이미지 및 마스크를 저장
    cv2.imwrite(os.path.join(image_dir, f"aug_{img_file}"), augmented_img)
    cv2.imwrite(os.path.join(mask_dir, f"aug_{mask_file}"), augmented_mask)
