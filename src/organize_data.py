import os
import shutil
import random

def organize_dataset():
    # Define paths based on project structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_images_dir = os.path.join(base_dir, 'data', 'all', 'images')
    all_labels_dir = os.path.join(base_dir, 'data', 'all', 'labels')
    output_dir = os.path.join(base_dir, 'data')
    
    # Class mapping for strawberry ripeness stages
    classes = {0: "partially_ripe", 1: "ripe", 2: "unripe"}

    # Create subdirectories for training and testing
    for split in ['train', 'test']:
        for cls_name in classes.values():
            os.makedirs(os.path.join(output_dir, split, cls_name), exist_ok=True)

    # Get all image files and shuffle for random distribution
    image_files = [f for f in os.listdir(all_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    # Split dataset: 80% for training, 20% for testing
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    def process_files(files, split_name):
        for img_name in files:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(all_labels_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    first_line = f.readline()
                    if not first_line: continue
                    
                    # Extract class ID from YOLO formatted label file
                    try:
                        class_id = int(first_line.split()[0])
                        class_name = classes.get(class_id, "unknown")
                        
                        # Copy image to its respective class folder
                        src_path = os.path.join(all_images_dir, img_name)
                        dst_path = os.path.join(output_dir, split_name, class_name, img_name)
                        shutil.copy2(src_path, dst_path)
                    except (ValueError, IndexError):
                        continue

    print("Processing and organizing dataset...")
    process_files(train_files, 'train')
    process_files(test_files, 'test')
    print("Dataset organization complete: data/train and data/test are ready.")

if __name__ == "__main__":
    organize_dataset()