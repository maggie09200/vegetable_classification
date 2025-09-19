import os
import random
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import numpy as np
import torch # 為了讓 AddGaussianNoise 類別能正確運作而導入

# 自定義添加高斯噪點的轉換
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# 定義資料增強的轉換流程
# 您可以根據需求調整參數
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 隨機水平翻轉
    transforms.RandomVerticalFlip(p=0.5),    # 隨機垂直翻轉
    transforms.RandomRotation(60),           # 在 (-45, 45) 度之間隨機旋轉
    transforms.ColorJitter(brightness=0.5, contrast=0.3), # 隨機調整亮度、對比度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)), # 隨機平移和縮放
    # transforms.Lambda(lambda img: add_gaussian_noise(img)) # 添加高斯噪點的另一種方法
])

def add_gaussian_noise(img, mean=0, std_dev=25):
    """
    為 PIL 圖片添加高斯噪點
    """
    img_array = np.array(img)
    noise = np.random.normal(mean, std_dev, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255) # 確保像素值在 0-255 範圍內
    return Image.fromarray(noisy_img_array.astype('uint8'))

def augment_images_in_folder(folder_path, target_count=500):
    """
    對指定資料夾中的圖片進行資料增強，直到達到目標數量。
    """
    if not os.path.isdir(folder_path):
        return
        
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    current_count = len(images)
    
    if current_count >= target_count:
        print(f"資料夾 '{os.path.basename(folder_path)}' 已有 {current_count} 張圖片，無需增強。")
        return
        
    print(f"資料夾 '{os.path.basename(folder_path)}' 中有 {current_count} 張圖片，開始增強至 {target_count} 張...")
    
    num_to_generate = target_count - current_count
    
    # 建立一個列表來追蹤新產生的圖片名稱，避免重複使用剛增強完的圖片
    existing_and_new_images = list(images)

    for i in range(num_to_generate):
        # 從現有圖片中隨機選擇一張作為基礎
        random_image_name = random.choice(existing_and_new_images)
        img_path = os.path.join(folder_path, random_image_name)
        
        try:
            with Image.open(img_path).convert("RGB") as img:
                # 應用 torchvision 的轉換
                augmented_img = transform(img)
                
                # 隨機決定是否添加高斯噪點
                if random.random() > 0.5:
                    augmented_img = add_gaussian_noise(augmented_img)

                # 定義新檔名
                original_name, extension = os.path.splitext(random_image_name)
                # 為了避免檔名衝突，使用更獨特的命名方式
                new_img_name = f"{os.path.splitext(images[i % len(images)])[0]}_aug_{i+1}{extension}"
                new_img_path = os.path.join(folder_path, new_img_name)
                
                # 儲存增強後的圖片
                augmented_img.save(new_img_path)
                existing_and_new_images.append(new_img_name)


        except Exception as e:
            print(f"處理圖片 {img_path} 時發生錯誤: {e}")

    print(f"完成！'{os.path.basename(folder_path)}' 資料夾現有 {len(os.listdir(folder_path))} 張圖片。")


# --- 主程式 (已根據新資料夾結構修改) ---
if __name__ == "__main__":
    base_dir = "veg_img_eng_new_aug50"  # 設定您的根資料夾
    
    # 檢查根資料夾是否存在
    if not os.path.exists(base_dir):
        print(f"根資料夾 '{base_dir}' 不存在。請檢查路徑。")
    else:
        print(f"--- 正在處理根資料夾: {base_dir} ---")
        
        # 直接遍歷根目錄下的所有蔬菜類別資料夾 (例如: "菠菜", "小白菜", "紅蘿蔔")
        for vegetable_folder in os.listdir(base_dir):
            vegetable_path = os.path.join(base_dir, vegetable_folder)
            
            # 確保我們只處理資料夾，忽略任何可能存在於根目錄的檔案
            if os.path.isdir(vegetable_path):
                augment_images_in_folder(vegetable_path, target_count=600)
            
    print("\n所有資料夾處理完畢。")