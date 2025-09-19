import cv2
import os
import numpy as np
import random

def augment_and_save(image, output_path, base_name, aug_count):
    """
    對單一張圖片進行傳統增強並儲存。
    （此函式保持不變）
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

    # 3. 隨機縮放
    if random.choice([True, False]):
        rows, cols, _ = image.shape
        scale = random.uniform(0.9, 1.1)
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        new_name = f"{base_name}_aug_{aug_count}_scale.jpg"
        new_scaled_image = np.zeros_like(image)
        if scale > 1.0:
            start_row = (scaled_image.shape[0] - rows) // 2
            start_col = (scaled_image.shape[1] - cols) // 2
            new_scaled_image = scaled_image[start_row:start_row+rows, start_col:start_col+cols]
        else:
            start_row = (rows - scaled_image.shape[0]) // 2
            start_col = (cols - scaled_image.shape[1]) // 2
            new_scaled_image[start_row:start_row+scaled_image.shape[0], start_col:start_col+scaled_image.shape[1]] = scaled_image
        cv2.imwrite(os.path.join(output_path, new_name), new_scaled_image)
        aug_count += 1
        
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

def perform_inter_class_mixup(class1_path, class1_original_files, all_class_dirs, aug_count):
    """
    對不同類別的兩張圖片進行Mixup增強，並存入主導類別的資料夾。

    Args:
        class1_path (str): 主導類別的資料夾路徑。
        class1_original_files (list): 主導類別中的原始圖片檔名列表。
        all_class_dirs (list): 所有類別資料夾的路徑列表。
        aug_count (int): 增強圖片的計數。

    Returns:
        int: 更新後的計數。
    """
    # 找出所有其他類別的資料夾
    other_class_dirs = [d for d in all_class_dirs if d != class1_path]
    if not other_class_dirs:
        return aug_count # 如果只有一個類別，無法執行跨類別 mixup

    # 隨機選擇一個其他類別
    class2_path = random.choice(other_class_dirs)
    class2_files = [f for f in os.listdir(class2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not class1_original_files or not class2_files:
        return aug_count # 如果任一資料夾為空，則跳過

    # 隨機選擇兩張來自不同類別的圖片
    img1_file = random.choice(class1_original_files)
    img2_file = random.choice(class2_files)
    
    img1 = cv2.imread(os.path.join(class1_path, img1_file))
    img2 = cv2.imread(os.path.join(class2_path, img2_file))

    if img1 is None or img2 is None:
        return aug_count

    # 確保兩張圖片尺寸相同，以第一張圖為基準調整第二張
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 設定 mixup 的混合比例 lambda，確保 class1 是主導
    # 讓 lambda 在 0.6 到 0.8 之間，確保 class1 的特徵更顯著
    lambda_val = random.uniform(0.6, 0.8)
    
    # 使用 cv2.addWeighted 進行圖片混合
    mixed_image = cv2.addWeighted(img1, lambda_val, img2_resized, 1 - lambda_val, 0)
    
    base_name1, _ = os.path.splitext(img1_file)
    class2_name = os.path.basename(class2_path)
    new_name = f"{base_name1}_mix_{class2_name}_aug_{aug_count}.jpg"
    
    # 將混合後的圖片儲存到 class1 的路徑下
    cv2.imwrite(os.path.join(class1_path, new_name), mixed_image)
    
    return aug_count + 1


def augment_images_in_directory(base_dir, target_count_per_class=600):
    """
    擴增指定資料夾中的圖片到目標數量，包含跨類別 Mixup。
    """
    if not os.path.isdir(base_dir):
        print(f"錯誤：找不到資料夾 '{base_dir}'")
        return

    # 先獲取所有類別的資料夾路徑
    all_class_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    all_class_dirs = [os.path.join(base_dir, name) for name in all_class_names]

    if len(all_class_dirs) < 2:
        print("警告：需要至少兩個類別才能執行跨類別 Mixup。將只執行傳統增強。")

    # 遍歷所有子資料夾（每個蔬菜類別）
    for class_name in all_class_names:
        class_path = os.path.join(base_dir, class_name)
        print(f"正在處理類別：{class_name}")

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(image_files)
        
        if current_count == 0:
            print(f"警告：在 '{class_path}' 中找不到任何圖片，跳過此類別。")
            continue

        print(f"找到 {current_count} 張原始圖片。")

        if current_count >= target_count_per_class:
            print(f"'{class_name}' 已有 {current_count} 張圖片，達到目標 {target_count_per_class}，無需擴增。")
            continue

        augment_needed = target_count_per_class - current_count
        original_image_files = list(image_files)
        
        augmented_count = 0
        while len(os.listdir(class_path)) < target_count_per_class:
            # 隨機選擇增強方式：50% 機率傳統增強，50% 機率 Mixup
            use_mixup = random.choice([True, False]) and len(all_class_dirs) >= 2

            if use_mixup:
                # 執行跨類別 Mixup
                augmented_count = perform_inter_class_mixup(class_path, original_image_files, all_class_dirs, augmented_count)
            else:
                # 執行傳統增強
                image_file = random.choice(original_image_files)
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path)
                if image is None: continue
                base_name, _ = os.path.splitext(image_file)
                augmented_count = augment_and_save(image, class_path, base_name, augmented_count)
        
        final_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"類別 '{class_name}' 處理完成。最終圖片數量：{final_count}\n")


# --- 主程式 ---
if __name__ == "__main__":
    # 設定你的資料根目錄
    data_directory = 'dataset_full_en_aug4/train'
    
    # 設定每個類別的目標圖片數量
    target_count = 600
    
    augment_images_in_directory(data_directory, target_count)
    print("所有類別的圖片擴增完成！")