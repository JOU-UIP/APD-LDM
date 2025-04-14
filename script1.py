import os
import random

# 数据集路径
data_dir = r'data/LOLdataset-256/test'
low_dir = os.path.join(data_dir, 'input')
high_dir = os.path.join(data_dir, 'GT')

# low_dir = r"data/underwater-test-dataset-U45--master/upload/U45-val/U45"
# high_dir = r"data/underwater-test-dataset-U45--master/upload/U45-val/FGAN"

# 输出的文件名
# output_file = r'data/underwater-test-dataset-U45--master/upload/U45-val/U45-val.txt'
output_file = r"data/LOLdataset-256/train/LOLdataset-256_val.txt"
# 获取 low 和 high 文件夹中的所有图片文件名
low_images = sorted(os.listdir(low_dir))
high_images = sorted(os.listdir(high_dir))




# 检查 low 和 high 文件夹中的图片数量
num_images = min(len(low_images), len(high_images))  # 使用较少的一组长度
print(f" number : {num_images}")
# 生成 unpaired_train.txt 文件
with open(output_file, 'w') as f:
    for i in range(num_images):
        low_image_path = os.path.join(low_dir, low_images[i])
        high_image_path = os.path.join(high_dir, high_images[i])
        f.write(f'{low_image_path} {high_image_path}\n')

print(f'Unpaired train file generated: {output_file}')
