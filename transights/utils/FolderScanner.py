from pathlib import Path
from typing import Union, List, Optional

class FolderScanner:
    @staticmethod
    def get_files(folders: Union[Path, str, List[Union[Path, str]]], extensions: Optional[Union[str, List[str]]]=None, recursive=False) -> List[Path]:
        """
        Get a list of files in the specified folders with optional extensions.

        Args:
            folders: A single folder path as a `Path` object or a string, or a list of folder paths.
            extensions: Optional. A single extension or a list of extensions. If provided, only files
                with matching extensions will be included.
            recursive: Wether subfolders are also scanned in a recursive fashion.

        Returns:
            A list of `Path` objects representing the files.

        Examples:
            # Retrieve all files in a folder
            files = FolderScanner.get_files(Path("path/to/folder"))

            # Retrieve files in multiple folders with specified extensions
            folders = [Path("path/to/folder1"), Path("path/to/folder2")]
            extensions = [".jpg", ".png"]
            files = FolderScanner.get_files(folders, extensions)

        """
        if isinstance(folders, (str, Path)):
            folders = [folders]  # Convert single folder path to list

        # TODO: refactor: too much duplicate code
        # scan for subfolders
        if recursive:
            subfolders = []
            for folder in folders:
                subfolders.extend([child for child in folder.glob('**/') if child.is_dir()])
            folders = subfolders

        if extensions is None:
            files = []
            for folder in folders:
                folder_path = Path(folder)
                for file_path in folder_path.iterdir():
                    if file_path.is_file():
                        files.append(file_path)
        else:
            if isinstance(extensions, str):
                extensions = [extensions]  # Convert single extension to list

            files = []
            for folder in folders:
                folder_path = Path(folder)
                for file_path in folder_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in extensions:
                        files.append(file_path)

        return files

    @staticmethod
    def get_image_files(folders: Union[Path, str, List[Union[Path, str]]]) -> List[Path]:
        """
        Get a list of image files (with default extensions) in the specified folders.

        Args:
            folders: A single folder path as a `Path` object or a string, or a list of folder paths.

        Returns:
            A list of `Path` objects representing the image files.

        Examples:
            # Retrieve image files in a folder using the default extensions
            image_files = FolderScanner.get_image_files(Path("path/to/folder"))

        """
        extensions = ['.jpg', '.jpeg', '.png', '.gif']
        return FolderScanner.get_files(folders, extensions)

    @staticmethod
    def get_csv_files(folders: Union[Path, str, List[Union[Path, str]]]) -> List[Path]:
        """
        Get a list of CSV files in the specified folders.

        Args:
            folders: A single folder path as a `Path` object or a string, or a list of folder paths.

        Returns:
            A list of `Path` objects representing the CSV files.

        Examples:
            # Retrieve CSV files in a folder
            csv_files = FolderScanner.get_csv_files(Path("path/to/folder"))

        """
        extensions = ['.csv']
        return FolderScanner.get_files(folders, extensions)
