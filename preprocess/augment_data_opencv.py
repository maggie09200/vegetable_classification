import cv2
import os
import numpy as np
import random

def augment_and_save(image, output_path, base_name, aug_count):
    """
    對單一張圖片進行增強並儲存。

    Args:
        image (numpy.ndarray): 輸入圖片。
        output_path (str): 儲存增強圖片的資料夾路徑。
        base_name (str): 原始圖片的檔名（不含副檔名）。
        aug_count (int): 增強圖片的計數。
    """
    # 1. 水平翻轉
    if random.choice([True, False]):
        flipped_image = cv2.flip(image, 1)
        new_name = f"{base_name}_aug_{aug_count}_flip.jpg"
        cv2.imwrite(os.path.join(output_path, new_name), flipped_image)
        aug_count += 1

    # 2. 隨機旋轉
    if random.choice([True, False]):
        rows, cols, _ = image.shape
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        new_name = f"{base_name}_aug_{aug_count}_rotate.jpg"
        cv2.imwrite(os.path.join(output_path, new_name), rotated_image)
        aug_count += 1

    # # 3. 隨機縮放
    # if random.choice([True, False]):
    #     rows, cols, _ = image.shape
    #     scale = random.uniform(0.9, 1.1)
    #     scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    #     new_name = f"{base_name}_aug_{aug_count}_scale.jpg"
    #     # 裁剪或填充以保持原始尺寸
    #     new_scaled_image = np.zeros_like(image)
    #     if scale > 1.0: # 如果放大，則中心裁剪
    #         start_row = (scaled_image.shape[0] - rows) // 2
    #         start_col = (scaled_image.shape[1] - cols) // 2
    #         new_scaled_image = scaled_image[start_row:start_row+rows, start_col:start_col+cols]
    #     else: # 如果縮小，則置中填充
    #         start_row = (rows - scaled_image.shape[0]) // 2
    #         start_col = (cols - scaled_image.shape[1]) // 2
    #         new_scaled_image[start_row:start_row+scaled_image.shape[0], start_col:start_col+scaled_image.shape[1]] = scaled_image
    #     cv2.imwrite(os.path.join(output_path, new_name), new_scaled_image)
    #     aug_count += 1
        
    # 4. 隨機平移
    if random.choice([True, False]):
        rows, cols, _ = image.shape
        tx = random.uniform(-0.1, 0.1) * cols
        ty = random.uniform(-0.1, 0.1) * rows
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, M, (cols, rows))
        new_name = f"{base_name}_aug_{aug_count}_translate.jpg"
        cv2.imwrite(os.path.join(output_path, new_name), translated_image)
        aug_count += 1

    return aug_count


def augment_images_in_directory(base_dir, target_count_per_class=600):
    """
    擴增指定資料夾中的圖片到目標數量。

    Args:
        base_dir (str): 包含各類別子資料夾的根目錄 (例如 'data/')。
        target_count_per_class (int): 每個類別的目標圖片數量。
    """
    if not os.path.isdir(base_dir):
        print(f"錯誤：找不到資料夾 '{base_dir}'")
        return

    # 遍歷所有子資料夾（每個蔬菜類別）
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            print(f"正在處理類別：{class_name}")

            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            current_count = len(image_files)
            
            if current_count == 0:
                print(f"警告：在 '{class_path}' 中找不到任何圖片，跳過此類別。")
                continue

            print(f"找到 {current_count} 張原始圖片。")

            # 如果目前圖片數量已達標，則跳過
            if current_count >= target_count_per_class:
                print(f"'{class_name}' 已有 {current_count} 張圖片，達到目標 {target_count_per_class}，無需擴增。")
                continue

            # 計算需要擴增的數量
            augment_needed = target_count_per_class - current_count
            
            # 遍歷原始圖片進行擴增
            augmented_count = 0
            while augmented_count < augment_needed:
                for image_file in image_files:
                    if augmented_count >= augment_needed:
                        break
                    
                    # 讀取圖片
                    image_path = os.path.join(class_path, image_file)
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"警告：無法讀取圖片 '{image_path}'，跳過。")
                        continue
                        
                    base_name, _ = os.path.splitext(image_file)
                    
                    # 進行隨機擴增並儲存
                    # 每次呼叫可能產生多張增強圖片
                    new_aug_count = augment_and_save(image, class_path, base_name, augmented_count)
                    augmented_count = new_aug_count

            final_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"類別 '{class_name}' 處理完成。最終圖片數量：{final_count}\n")


# --- 主程式 ---
if __name__ == "__main__":
    # 設定你的資料根目錄
    data_directory = '../dataset_full_en_aug7_56_new/train'
    
    # 設定每個類別的目標圖片數量
    target_count = 600
    
    augment_images_in_directory(data_directory, target_count)
    print("所有類別的圖片擴增完成！")