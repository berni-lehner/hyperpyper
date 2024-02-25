import os
import zipfile
import requests
from pathlib import Path
import gdown
from bs4 import BeautifulSoup

class GDriveDownloader:
    """
    Example:
    
    url = 'https://drive.google.com/file/d/1LK1VyjvQfA3qUG8LGue0IBOy0Sja2GJb/view?usp=sharing'
    destination = Path("destination_folder")
    GDriveDownloader.download_and_unpack(url, destination)
    """
    @staticmethod
    def download_and_unpack(self,
                            url: str,
                            dst: Path,
                            cache=True,
                            remove_org_file=False):
        # Check if the destination directory exists, if not, create it
        if not dst.exists():
            dst.mkdir(parents=True)

        # Extract file name from URL
        file_name = GDriveDownloader._get_file_name(url)

        # Check if the file already exists in the destination path
        file_path = dst / file_name
        if cache and file_path.exists():
            print(f"File '{file_name}' already exists. Skipping download.")
            return

        # Download file from Google Drive
        print(f"Downloading file from {url}...")
        gdown.download(url=url, output=str(file_path), quiet=False, fuzzy=True)

        # Check if the downloaded file is a zip file, if so, extract it
        if file_name.endswith('.zip'):
            print("Extracting zip file...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(str(dst))
        
        # Remove the original zip file if needed
        if remove_org_file: 
            print("Remove the original zip file...")
            os.remove(file_path)

        print(f"Downloading and extracing complete...")
        

    @staticmethod
    def _get_file_name(self, url: str):
        # Send GET request
        response = requests.get(url)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the title element which contains the file name
        title_element = soup.find('title')

        # Extract the file name
        file_name = title_element.text.split(' - ')[0]

        return file_name