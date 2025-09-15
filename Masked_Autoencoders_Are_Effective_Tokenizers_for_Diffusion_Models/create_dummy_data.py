import os
from PIL import Image
import numpy as np

def create_dummy_imagenet(base_path):
    """Creates a minimal, fake ImageNet structure."""
    print(f"Creating dummy ImageNet structure at {base_path}")
    
    # Define paths
    raw_train_dir = os.path.join(base_path, 'train')
    raw_val_dir = os.path.join(base_path, 'val')
    
    # Create directories
    class_dir_1 = os.path.join(raw_train_dir, 'n01440764') # A real ImageNet class name
    class_dir_2 = os.path.join(raw_train_dir, 'n01443537')
    os.makedirs(class_dir_1, exist_ok=True)
    os.makedirs(class_dir_2, exist_ok=True)
    os.makedirs(raw_val_dir, exist_ok=True)
    
    # Create dummy images
    def create_fake_image(path, size=(256, 256)):
        if not os.path.exists(path):
            img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(path)

    # Create a few images for each class
    create_fake_image(os.path.join(class_dir_1, 'dummy_train_1_1.JPEG'))
    create_fake_image(os.path.join(class_dir_1, 'dummy_train_1_2.JPEG'))
    create_fake_image(os.path.join(class_dir_2, 'dummy_train_2_1.JPEG'))
    create_fake_image(os.path.join(class_dir_2, 'dummy_train_2_2.JPEG'))
    
    # Create validation images
    val_img_1_path = os.path.join(raw_val_dir, 'ILSVRC2012_val_00000001.JPEG')
    val_img_2_path = os.path.join(raw_val_dir, 'ILSVRC2012_val_00000002.JPEG')
    create_fake_image(val_img_1_path)
    create_fake_image(val_img_2_path)
    
    print("Dummy structure created.")

if __name__ == '__main__':
    # The user requested the dataset to be in ../datasets/imagenet
    # The current working directory is /root/PapersImplementation/Masked_Autoencoders_Are_Effective_Tokenizers_for_Diffusion_Models
    # So the relative path is correct.
    target_dir = '/root/PapersImplementation/datasets/imagenet'
    create_dummy_imagenet(target_dir)
