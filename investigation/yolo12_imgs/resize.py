import os
from PIL import Image

input_folder = r"C:\sem8\VKR\DIPLOM_SRC\investigation\yolo12_imgs\test_diff_sizes_imgs"
output_base_folder = r"C:\sem8\VKR\DIPLOM_SRC\investigation\yolo12_imgs\test_diff_sizes_imgs_resized"

original_height = 2048
min_height = 8
step = 10  # шаг: 2048, 2038, 2028, ..., 8

target_heights = list(range(original_height, min_height - 1, -step))

print(f"Целевые высоты: {target_heights[:5]}...{target_heights[-5:]}")
print(f"Всего размеров: {len(target_heights)}")

# Создаём основную папку
os.makedirs(output_base_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        number = filename.split('-')[0]
        input_path = os.path.join(input_folder, filename)
        
        with Image.open(input_path) as img:
            orig_width, orig_height = img.size
            
            for target_height in target_heights:
                if target_height > orig_height:
                    continue
                
                new_width = int(round(orig_width * (target_height / orig_height)))
                if new_width < 1 or target_height < 1:
                    continue
                
                resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                
                # Создаём подпапку для этой высоты
                height_folder = os.path.join(output_base_folder, f"height_{target_height}")
                os.makedirs(height_folder, exist_ok=True)
                
                extension = os.path.splitext(filename)[1]
                output_filename = f"{number}-{target_height}{extension}"
                output_path = os.path.join(height_folder, output_filename)
                
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    resized_img.save(output_path, 'JPEG', quality=85, optimize=True)
                else:
                    resized_img.save(output_path, optimize=True)
            
            print(f"Обработан: {filename}")

print(f"Готово! Результаты в {output_base_folder}")

# Проверяем создание папок
if os.path.exists(output_base_folder):
    subfolders = [d for d in os.listdir(output_base_folder) if os.path.isdir(os.path.join(output_base_folder, d))]
    print(f"Создано подпапок: {len(subfolders)}")
    print(f"Примеры: {subfolders[:5]}")