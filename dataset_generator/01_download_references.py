import os
import time
import requests
import urllib.parse
from tqdm import tqdm
import json
import random
from typing import List, Dict, Set, Tuple

class ScryfallDatasetGenerator:
    """Generator for MTG card datasets using Scryfall API"""
    
    def __init__(self):
        self.base_url = "https://api.scryfall.com/cards/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTG-Dataset-Generator/1.0'
        })
        self.rate_limit_delay = 0.2  # 100ms between requests
        
        # Dataset configuration
        self.train_total = 2000
        self.test_total = 500
        self.full_art_ratio = 0.25  # 1/4 of images should be full_art
        
        # Calculate exact counts
        self.train_full_art = int(self.train_total * self.full_art_ratio)  # 500
        self.train_normal = self.train_total - self.train_full_art  # 1500
        self.test_full_art = int(self.test_total * self.full_art_ratio)  # 125
        self.test_normal = self.test_total - self.test_full_art  # 375
        
        # Track downloaded cards to avoid duplicates
        self.used_card_names: Set[str] = set()
        
        print(f"Dataset Configuration:")
        print(f"Training: {self.train_total} images ({self.train_full_art} full_art, {self.train_normal} normal)")
        print(f"Test: {self.test_total} images ({self.test_full_art} full_art, {self.test_normal} normal)")

    def create_directories(self):
        """Create necessary directories for datasets"""
        directories = [
            'references/train',
            'references/test'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

    def build_api_url(self, page: int = 1) -> str:
        """Build Scryfall API URL with proper parameters"""
        params = {
            'format': 'json',
            'include_extras': 'false',
            'include_multilingual': 'false',
            'include_variations': 'false',
            'order': 'cmc',
            'page': str(page),
            'q': '(game:paper)',
            'unique': 'prints'
        }
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.base_url}?{query_string}"

    def fetch_cards_page(self, url: str, retries: int = 3) -> Tuple[List[Dict], str]:
        """Fetch a page of cards from Scryfall API with retries"""
        for attempt in range(retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                cards = data.get('data', [])
                next_page = data.get('next_page', None)
                
                return cards, next_page
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching cards (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)  # Wait 1 second before retrying
                else:
                    print("Max retries reached. Skipping page.")
        
        return [], None

    def filter_valid_cards(self, cards: List[Dict]) -> List[Dict]:
        """Filter cards that meet our criteria"""
        valid_cards = []
        
        for card in cards:
            # Check required criteria
            if (card.get('image_status') == 'highres_scan' and 
                card.get('highres_image', False) and
                card.get('image_uris') and
                card.get('image_uris', {}).get('png') and
                card.get('name') not in self.used_card_names):
                
                valid_cards.append(card)
        
        return valid_cards

    def categorize_cards(self, cards: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Separate cards into full_art and normal categories"""
        full_art_cards = []
        normal_cards = []
        
        for card in cards:
            if card.get('full_art', False):
                full_art_cards.append(card)
            else:
                normal_cards.append(card)
        
        return full_art_cards, normal_cards

    def download_image(self, url: str, filepath: str, retries: int = 3) -> bool:
        """Download an image from URL to filepath with retries"""
        for attempt in range(retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url} (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)  # Wait 1 second before retrying
                else:
                    print(f"Max retries reached for {url}. Skipping file.")
        
        return False

    def get_filename(self, card: Dict, is_full_art: bool) -> str:
        """Generate filename for card image"""
        card_id = card.get('id', 'unknown')
        prefix = 'full_art' if is_full_art else 'normal'
        return f"{prefix}_{card_id}.png"

    def collect_cards(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Collect all required cards from Scryfall API"""
        print("Collecting cards from Scryfall API...")
        
        # Collections for each category
        train_full_art = []
        train_normal = []
        test_full_art = []
        test_normal = []
        
        # Start fetching cards
        url = self.build_api_url(1)
        page = 1
        
        with tqdm(desc="Fetching cards", unit="pages") as pbar:
            while url and (len(train_full_art) < self.train_full_art or 
                          len(train_normal) < self.train_normal or
                          len(test_full_art) < self.test_full_art or
                          len(test_normal) < self.test_normal):
                
                cards, next_url = self.fetch_cards_page(url)
                if not cards:
                    break
                
                valid_cards = self.filter_valid_cards(cards)
                full_art_cards, normal_cards = self.categorize_cards(valid_cards)
                
                # Shuffle to randomize selection
                random.shuffle(full_art_cards)
                random.shuffle(normal_cards)
                
                # Distribute cards to train/test sets
                for card in full_art_cards:
                    if len(train_full_art) < self.train_full_art:
                        train_full_art.append(card)
                        self.used_card_names.add(card['name'])
                    elif len(test_full_art) < self.test_full_art:
                        test_full_art.append(card)
                        self.used_card_names.add(card['name'])
                
                for card in normal_cards:
                    if len(train_normal) < self.train_normal:
                        train_normal.append(card)
                        self.used_card_names.add(card['name'])
                    elif len(test_normal) < self.test_normal:
                        test_normal.append(card)
                        self.used_card_names.add(card['name'])
                
                pbar.set_postfix({
                    'Train FA': f"{len(train_full_art)}/{self.train_full_art}",
                    'Train N': f"{len(train_normal)}/{self.train_normal}",
                    'Test FA': f"{len(test_full_art)}/{self.test_full_art}",
                    'Test N': f"{len(test_normal)}/{self.test_normal}"
                })
                pbar.update(1)
                
                url = next_url
                page += 1
        
        print(f"\nCard collection complete:")
        print(f"Train full_art: {len(train_full_art)}, Train normal: {len(train_normal)}")
        print(f"Test full_art: {len(test_full_art)}, Test normal: {len(test_normal)}")
        
        return train_full_art, train_normal, test_full_art, test_normal

    def download_dataset(self, cards: List[Dict], directory: str, category: str, is_full_art: bool):
        """Download images for a specific dataset category"""
        print(f"\nDownloading {category} images...")
        
        successful_downloads = 0
        failed_downloads = 0
        
        with tqdm(total=len(cards), desc=f"Downloading {category}") as pbar:
            for card in cards:
                image_url = card['image_uris']['png']
                filename = self.get_filename(card, is_full_art)
                filepath = os.path.join(directory, filename)
                
                if self.download_image(image_url, filepath):
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                
                pbar.set_postfix({
                    'Success': successful_downloads,
                    'Failed': failed_downloads
                })
                pbar.update(1)
        
        print(f"{category} download complete: {successful_downloads} successful, {failed_downloads} failed")

    def generate_datasets(self):
        """Main method to generate both training and test datasets"""
        print("Starting MTG card dataset generation...")
        
        # Create directories
        self.create_directories()
        
        # Collect all required cards
        train_full_art, train_normal, test_full_art, test_normal = self.collect_cards()
        
        # Download training dataset
        print("\n" + "="*50)
        print("DOWNLOADING TRAINING DATASET")
        print("="*50)
        self.download_dataset(train_full_art, 'references/train', 'Train Full Art', True)
        self.download_dataset(train_normal, 'references/train', 'Train Normal', False)
        
        # Download test dataset
        print("\n" + "="*50)
        print("DOWNLOADING TEST DATASET")
        print("="*50)
        self.download_dataset(test_full_art, 'references/test', 'Test Full Art', True)
        self.download_dataset(test_normal, 'references/test', 'Test Normal', False)
        
        print("\n" + "="*50)
        print("DATASET GENERATION COMPLETE!")
        print("="*50)
        print(f"Training dataset: references/train/ ({self.train_total} images)")
        print(f"Test dataset: references/test/ ({self.test_total} images)")


def main():
    """Main execution function"""
    try:
        generator = ScryfallDatasetGenerator()
        generator.generate_datasets()
        
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
