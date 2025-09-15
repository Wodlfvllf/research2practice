import torchvision.transforms as transforms

class DataTransforms:
    """
    A class to provide data transformations for ImageNet.
    """
    def __init__(self, img_size=256):
        # Normalization parameters for ImageNet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # Transformations for the training set
        self.train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Transformations for the validation/test set
        self.val_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
