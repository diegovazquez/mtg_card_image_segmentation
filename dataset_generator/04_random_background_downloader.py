import os
import time
import requests
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
from typing import Set, Tuple

class LoremPicsumBackgroundGenerator:
    """Generator for random background images using Lorem Picsum API"""
    
    def __init__(self):
        self.base_url = "https://picsum.photos"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTG-Background-Generator/1.0'
        })
        self.rate_limit_delay = 0.1  # 100ms between requests to be respectful
        
        # Image configuration
        self.image_width = 480
        self.image_height = 640
        self.image_format = "jpg"
        self.mask_format = "png"
        
        # Dataset configuration - original targets
        self.train_total_original = 800
        self.test_total_original = 200
        
        # Track downloaded image IDs to avoid duplicates
        self.used_image_ids: Set[str] = set()
        
        # Count existing images and adjust totals
        existing_train, existing_test = self.count_existing_images()
        self.train_existing = existing_train
        self.test_existing = existing_test
        self.train_total = max(0, self.train_total_original - existing_train)
        self.test_total = max(0, self.test_total_original - existing_test)
        
        print(f"Lorem Picsum Background Dataset Configuration:")
        print(f"Image dimensions: {self.image_width}x{self.image_height} (portrait)")
        print(f"Training images: {self.train_total_original} ({existing_train} existing, {self.train_total} to download)")
        print(f"Test images: {self.test_total_original} ({existing_test} existing, {self.test_total} to download)")
        print(f"Total images to download: {self.train_total + self.test_total}")

    def count_existing_images(self) -> Tuple[int, int]:
        """Count existing Lorem Picsum images in train and test directories"""
        train_count = 0
        test_count = 0
        
        # Define directories to check
        train_images_dir = 'dataset/train/images'
        test_images_dir = 'dataset/test/images'
        
        # Count training images
        if os.path.exists(train_images_dir):
            for filename in os.listdir(train_images_dir):
                if filename.startswith('lorem_picsum_') and filename.endswith(f'.{self.image_format}'):
                    train_count += 1
                    # Extract image ID and add to used_image_ids to prevent re-downloading
                    image_id = filename.replace('lorem_picsum_', '').replace(f'.{self.image_format}', '')
                    self.used_image_ids.add(image_id)
        
        # Count test images
        if os.path.exists(test_images_dir):
            for filename in os.listdir(test_images_dir):
                if filename.startswith('lorem_picsum_') and filename.endswith(f'.{self.image_format}'):
                    test_count += 1
                    # Extract image ID and add to used_image_ids to prevent re-downloading
                    image_id = filename.replace('lorem_picsum_', '').replace(f'.{self.image_format}', '')
                    self.used_image_ids.add(image_id)
        
        return train_count, test_count

    def create_directories(self):
        """Create necessary directories for datasets"""
        directories = [
            'dataset/train/images',
            'dataset/train/masks',
            'dataset/test/images', 
            'dataset/test/masks'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")

    def get_random_image_url(self) -> str:
        """Generate URL for random image from Lorem Picsum"""
        return f"{self.base_url}/{self.image_width}/{self.image_height}"

    def extract_image_id_from_response(self, response: requests.Response) -> str:
        """Extract image ID from Lorem Picsum response"""
        # Lorem Picsum redirects to the actual image URL which contains the ID
        # Example: https://picsum.photos/id/237/480/640.jpg
        final_url = response.url
        
        # Try to extract ID from the final URL
        if '/id/' in final_url:
            # Extract ID from URL like: https://picsum.photos/id/237/480/640.jpg
            parts = final_url.split('/id/')
            if len(parts) > 1:
                id_part = parts[1].split('/')[0]
                return id_part
        
        # Fallback: generate a unique ID based on content hash
        import hashlib
        content_hash = hashlib.md5(response.content).hexdigest()[:8]
        return f"hash_{content_hash}"

    def download_image(self, image_dir: str, mask_dir: str, retries: int = 3) -> Tuple[bool, str]:
        """Download a random image and create corresponding black mask"""
        for attempt in range(retries):
            try:
                time.sleep(self.rate_limit_delay)
                
                # Get random image
                url = self.get_random_image_url()
                response = self.session.get(url, timeout=30, allow_redirects=True)
                response.raise_for_status()

                # Extract image ID
                image_id = self.extract_image_id_from_response(response)
                
                # Check if we already have this image
                if image_id in self.used_image_ids:
                    #print(f"Image {image_id} already exists, skipping...")
                    continue  # Try again with a different image
                
                # Generate filenames
                image_filename = f"lorem_picsum_{image_id}.{self.image_format}"
                mask_filename = f"lorem_picsum_{image_id}.{self.mask_format}"
                
                image_path = os.path.realpath(os.path.join(image_dir, image_filename))
                mask_path = os.path.realpath(os.path.join(mask_dir, mask_filename))
                
                print(f"Downloaded image {image_id} successfully: {image_path}")

                # Save the downloaded image
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify image dimensions and format
                try:
                    with Image.open(image_path) as img:
                        if img.size != (self.image_width, self.image_height):
                            print(f"Warning: Image {image_id} has unexpected dimensions: {img.size}")
                except Exception as e:
                    print(f"Warning: Could not verify image {image_id}: {e}")
                
                # Create corresponding black mask
                self.create_black_mask(mask_path)
                
                # Mark this image ID as used
                self.used_image_ids.add(image_id)
                
                return True, image_id
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading image (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(0.05)  # Wait x seconds before retrying
                else:
                    print("Max retries reached. Skipping this image.")
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(0.05)
                else:
                    print("Max retries reached due to unexpected error.")
        
        return False, ""

    def create_black_mask(self, mask_path: str):
        """Create a completely black mask image"""
        # Create a black image with the same dimensions
        black_image = Image.new('RGB', (self.image_width, self.image_height), color='black')
        
        # Save as PNG
        black_image.save(mask_path, 'PNG')

    def download_dataset(self, image_dir: str, mask_dir: str, total_images: int, dataset_name: str):
        """Download images and create masks for a specific dataset"""
        if total_images == 0:
            print(f"\n{dataset_name} dataset: All images already exist, skipping download.")
            return 0
        
        print(f"\nDownloading {dataset_name} dataset...")
        
        successful_downloads = 0
        failed_downloads = 0
        
        with tqdm(total=total_images, desc=f"Downloading {dataset_name}") as pbar:
            while successful_downloads < total_images:
                success, image_id = self.download_image(image_dir, mask_dir)
                
                if success:
                    successful_downloads += 1
                    pbar.set_postfix({
                        'Success': successful_downloads,
                        'Failed': failed_downloads,
                        'Last ID': image_id
                    })
                    pbar.update(1)
                else:
                    failed_downloads += 1
                    pbar.set_postfix({
                        'Success': successful_downloads,
                        'Failed': failed_downloads,
                        'Skipped': failed_downloads
                    })
                
                # Safety check to prevent infinite loops
                if failed_downloads > total_images * 0.5:  # If more than 50% failed
                    print(f"\nWarning: High failure rate detected. Stopping {dataset_name} download.")
                    break
        
        print(f"{dataset_name} download complete: {successful_downloads} successful, {failed_downloads} failed")
        return successful_downloads

    def generate_datasets(self):
        """Main method to generate both training and test datasets"""
        print("Starting Lorem Picsum background dataset generation...")
        
        # Create directories
        self.create_directories()
        
        # Download training dataset
        print("\n" + "="*60)
        print("DOWNLOADING TRAINING DATASET")
        print("="*60)
        train_success = self.download_dataset(
            'dataset/train/images',
            'dataset/train/masks',
            self.train_total,
            'Training'
        )
        
        # Download test dataset
        print("\n" + "="*60)
        print("DOWNLOADING TEST DATASET")
        print("="*60)
        test_success = self.download_dataset(
            'dataset/test/images',
            'dataset/test/masks',
            self.test_total,
            'Test'
        )
        
        # Final summary
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"Training dataset: {self.train_existing + train_success}/{self.train_total_original} images ({self.train_existing} existing, {train_success} newly downloaded)")
        print(f"Test dataset: {self.test_existing + test_success}/{self.test_total_original} images ({self.test_existing} existing, {test_success} newly downloaded)")
        print(f"Total images in dataset: {self.train_existing + self.test_existing + train_success + test_success}")
        print(f"Total newly downloaded: {train_success + test_success}")
        print(f"Images saved to: dataset/train/images/ and dataset/test/images/")
        print(f"Black masks saved to: dataset/train/masks/ and dataset/test/masks/")

def main():
    """Main execution function"""
    try:
        generator = LoremPicsumBackgroundGenerator()
        generator.generate_datasets()
        
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
