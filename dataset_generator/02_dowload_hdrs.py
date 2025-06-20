import os
import time
import requests
import urllib.parse
from tqdm import tqdm
import json
import random
from typing import List, Dict, Optional, Tuple

class PolyhavenHDRIDownloader:
    """Downloader for HDRI assets from Polyhaven API"""
    
    def __init__(self):
        self.base_url = "https://api.polyhaven.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HDRI-Dataset-Generator/1.0'
        })
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Download configuration
        self.target_count = 20
        self.target_category = "indoor"
        self.target_resolution = "8k"
        
        # Supported HDRI formats (prioritized)
        self.preferred_formats = ['hdr', 'exr']
        
        print(f"HDRI Download Configuration:")
        print(f"Target: {self.target_count} HDRIs")
        print(f"Category: {self.target_category}")
        print(f"Resolution: {self.target_resolution}")

    def create_directories(self):
        """Create necessary directories for HDRI storage"""
        # Use realpath for cross-platform compatibility
        script_dir = os.path.dirname(os.path.realpath(__file__))
        hdri_dir = os.path.join(script_dir, 'hdri')
        
        os.makedirs(hdri_dir, exist_ok=True)
        print(f"Created directory: {os.path.realpath(hdri_dir)}")
        
        return os.path.realpath(hdri_dir)

    def get_indoor_hdris(self) -> List[Dict]:
        """Fetch all indoor HDRI assets from Polyhaven API"""
        print("Fetching indoor HDRIs from Polyhaven API...")
        
        url = f"{self.base_url}/assets"
        params = {
            'type': 'hdris',
            'categories': self.target_category
        }
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            hdris = []
            
            # Convert the response dict to a list of HDRIs with their IDs
            for hdri_id, hdri_data in data.items():
                hdri_info = hdri_data.copy()
                hdri_info['id'] = hdri_id
                hdris.append(hdri_info)
            
            print(f"Found {len(hdris)} indoor HDRIs")
            return hdris
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching HDRIs: {e}")
            return []

    def get_hdri_files(self, hdri_id: str) -> Optional[Dict]:
        """Get file information for a specific HDRI"""
        url = f"{self.base_url}/files/{hdri_id}"
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching files for HDRI {hdri_id}: {e}")
            return None

    def find_8k_download_url(self, files_data: Dict) -> Optional[Tuple[str, str, int]]:
        """Find 8K download URL from files data"""
        hdri_files = files_data.get('hdri', {})
        
        # Look for 8k resolution
        if self.target_resolution in hdri_files:
            resolution_files = hdri_files[self.target_resolution]
            
            # Try preferred formats in order
            for format_name in self.preferred_formats:
                if format_name in resolution_files:
                    file_info = resolution_files[format_name]
                    if 'url' in file_info:
                        return (
                            file_info['url'],
                            format_name,
                            file_info.get('size', 0)
                        )
        
        return None

    def filter_8k_hdris(self, hdris: List[Dict]) -> List[Dict]:
        """Filter HDRIs that have 8K versions available"""
        print("Filtering HDRIs with 8K versions...")
        
        available_8k_hdris = []
        
        with tqdm(total=len(hdris), desc="Checking 8K availability") as pbar:
            for hdri in hdris:
                files_data = self.get_hdri_files(hdri['id'])
                if files_data:
                    download_info = self.find_8k_download_url(files_data)
                    if download_info:
                        hdri['download_url'] = download_info[0]
                        hdri['file_format'] = download_info[1]
                        hdri['file_size'] = download_info[2]
                        available_8k_hdris.append(hdri)
                
                pbar.update(1)
        
        print(f"Found {len(available_8k_hdris)} HDRIs with 8K versions")
        return available_8k_hdris

    def select_random_hdris(self, hdris: List[Dict]) -> List[Dict]:
        """Randomly select HDRIs for download"""
        available_count = len(hdris)
        download_count = min(self.target_count, available_count)
        
        if available_count < self.target_count:
            print(f"Warning: Only {available_count} 8K indoor HDRIs available, downloading all of them")
        
        selected_hdris = random.sample(hdris, download_count)
        
        print(f"Selected {len(selected_hdris)} HDRIs for download:")
        for hdri in selected_hdris:
            size_mb = hdri['file_size'] / (1024 * 1024) if hdri['file_size'] > 0 else 0
            print(f"  - {hdri['name']} ({hdri['file_format'].upper()}, {size_mb:.1f}MB)")
        
        return selected_hdris

    def get_safe_filename(self, hdri: Dict) -> str:
        """Generate safe filename for HDRI"""
        hdri_id = hdri['id']
        file_format = hdri['file_format']
        
        # Create safe filename using HDRI ID
        safe_name = hdri_id.replace(' ', '_').replace('/', '_').replace('\\', '_')
        return f"{safe_name}.{file_format}"

    def download_hdri(self, hdri: Dict, output_dir: str) -> bool:
        """Download individual HDRI with progress tracking"""
        filename = self.get_safe_filename(hdri)
        filepath = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            return True
        
        try:
            response = self.session.get(hdri['download_url'], stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {hdri['name'][:30]}...",
                    leave=False
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✓ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            # Clean up partial download
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return False

    def download_hdris(self, hdris: List[Dict], output_dir: str):
        """Download all selected HDRIs"""
        print(f"\nStarting download of {len(hdris)} HDRIs...")
        print("="*60)
        
        successful_downloads = 0
        failed_downloads = 0
        
        with tqdm(total=len(hdris), desc="Overall Progress", unit="HDRIs") as overall_pbar:
            for hdri in hdris:
                if self.download_hdri(hdri, output_dir):
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                
                overall_pbar.set_postfix({
                    'Success': successful_downloads,
                    'Failed': failed_downloads
                })
                overall_pbar.update(1)
                
                # Small delay between downloads
                time.sleep(self.rate_limit_delay)
        
        print("="*60)
        print(f"Download complete!")
        print(f"Successful: {successful_downloads}")
        print(f"Failed: {failed_downloads}")
        print(f"Total HDRIs: {successful_downloads + failed_downloads}")

    def download_indoor_hdris(self):
        """Main method to download indoor HDRIs"""
        print("Starting indoor HDRI download process...")
        print("="*60)
        
        try:
            # Create output directory
            output_dir = self.create_directories()
            
            # Get all indoor HDRIs
            all_hdris = self.get_indoor_hdris()
            if not all_hdris:
                print("No indoor HDRIs found. Exiting.")
                return
            
            # Filter for 8K versions
            hdris_8k = self.filter_8k_hdris(all_hdris)
            if not hdris_8k:
                print("No indoor HDRIs with 8K versions found. Exiting.")
                return
            
            # Select random HDRIs
            selected_hdris = self.select_random_hdris(hdris_8k)
            
            # Download selected HDRIs
            self.download_hdris(selected_hdris, output_dir)
            
            print("\n" + "="*60)
            print("HDRI DOWNLOAD PROCESS COMPLETE!")
            print("="*60)
            print(f"HDRIs saved to: {output_dir}")
            
        except KeyboardInterrupt:
            print("\nDownload interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

def main():
    """Main execution function"""
    try:
        downloader = PolyhavenHDRIDownloader()
        downloader.download_indoor_hdris()
        
    except KeyboardInterrupt:
        print("\nHDRI download interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
