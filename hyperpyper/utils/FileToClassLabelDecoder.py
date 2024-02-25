from pathlib import Path
from typing import List, Union
import numpy as np

class FileToClassLabelDecoder:
    """
    Extract class labels from folder paths.

    Args:
        auto_typing (bool): If True, automatically determine whether class labels should be converted to integers based on unique labels. Default is True.
        parts_idx (int): The index to extract from the parts of the file path. Default is -2.

    Attributes:
        auto_typing (bool): If True, automatically determine whether class labels should be converted to integers based on unique labels.
        parts_idx (int): The index to extract from the parts of the file path.
    """

    def __init__(self, auto_typing: bool=True, parts_idx: int=-2):
        self.auto_typing: bool = auto_typing
        self.parts_idx = parts_idx


    def __call__(self, files: List[Union[str, Path]]) -> np.ndarray:
        """
        Extracts class labels from folder paths.

        Args:
            files (List[Union[str, Path]]): List of file paths.

        Returns:
            np.ndarray: Array containing extracted class labels.
        """
        class_labels = [Path(file).parts[self.parts_idx] for file in files]
        
        if self.auto_typing:
            unique_labels = np.unique(class_labels)
            if all(label.isdigit() for label in unique_labels):
                class_labels = [int(label) for label in class_labels]

        return np.array(class_labels)