import os
import sys
import shutil

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

def move_images(source_folder, target_folder):
    """
    将 source_folder 中所有子文件夹的图片移动到 target_folder 中。

    :param source_folder: 包含子文件夹的主文件夹路径
    :param target_folder: 目标主文件夹路径
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有子文件夹和文件
    for root, _, files in os.walk(source_folder):
        for file in files:
            # 检查文件是否是图片（根据扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # 构建源文件的完整路径
                source_path = os.path.join(root, file)
                # 构建目标文件的完整路径
                target_path = os.path.join(target_folder, file)
                
                # 如果目标文件夹中已经存在同名文件，则重命名
                if os.path.exists(target_path):
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_path):
                        new_file_name = f"{base}_{counter}{extension}"
                        target_path = os.path.join(target_folder, new_file_name)
                        counter += 1
                
                # 移动文件
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} -> {target_path}")

train_folder = r"./train"  
target_folder_1 = r"./train"  
test_folder = r"./test"  
target_folder_2 = r"./test"  
move_images(train_folder, target_folder_1)
move_images(test_folder, target_folder_2)