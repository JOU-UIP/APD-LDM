import os
from PIL import Image
from torchvision import transforms

# 定义转换：将图像调整为 512x512
resize_transform = transforms.Resize((512, 512))
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# 输入和输出文件夹路径
input_folder = r'data/LOLdataset/eval15/low'  # 替换为你的输入文件夹路径
output_folder = r'data/LOL-v1-512/low'  # 替换为你的输出文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 遍历输入文件夹中的所有 .png 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 构造完整的文件路径
        img_path = os.path.join(input_folder, filename)
        # print(f"Processing file: {img_path}")
        # 打开图像文件
        img = Image.open(img_path)
        
        # 转换为张量并调整大小
        img_resized = resize_transform(img)
        
        # 将调整大小后的图像转换为 PIL 格式
        img_resized_pil = to_pil(to_tensor(img_resized))
        
        # 保存调整后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        img_resized_pil.save(output_path)
        
        print(f"Resized and saved: {output_path}")

print("All images have been resized and saved.")
