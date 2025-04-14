import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        初始化数据集
        :param folder_path: 图像文件夹路径
        :param transform: 图像预处理变换
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取指定索引的图像
        :param idx: 索引
        :return: 图像张量
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # 保证图像是RGB格式
        if self.transform:
            image = self.transform(image)
        return image , image_path


# 测试加载器是否工作正常
if __name__ == "__main__":
    # 定义测试集数据变换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 文件夹路径
    folder_path = "../data/challenging-60"

    # 创建数据集和数据加载器
    test_dataset = ImageFolderDataset(folder_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    for batch_idx, images in enumerate(test_loader):
        print(f"Batch {batch_idx}: {images.size()}")
