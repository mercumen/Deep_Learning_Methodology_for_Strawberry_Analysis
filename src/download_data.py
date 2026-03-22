import os
import zipfile
import subprocess

def setup_dataset():
    # Kaggle dataset identifier
    DATASET_SLUG = "mahyeks/multi-class-strawberry-ripeness-detection-dataset" 
    
    # Define the target data directory relative to the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading {DATASET_SLUG} from Kaggle...")
    
    try:
        # Execute Kaggle API command to download the dataset
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", data_dir], check=True)
        
        # Identify the downloaded zip file path
        zip_filename = DATASET_SLUG.split('/')[-1] + ".zip"
        zip_filepath = os.path.join(data_dir, zip_filename)
        
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Cleanup: Remove the zip file after extraction to save space
        os.remove(zip_filepath)
        print("Dataset successfully downloaded, extracted, and organized.")
        
    except FileNotFoundError:
        print("Error: Kaggle CLI not found. Please run 'pip install kaggle'.")
    except subprocess.CalledProcessError:
        print("Error: Download failed. Ensure kaggle.json is in your .kaggle folder.")

if __name__ == "__main__":
    setup_dataset()