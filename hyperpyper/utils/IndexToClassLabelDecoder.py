from typing import List

class IndexToClassLabelDecoder:
    """
    Decoder class to convert indices to class labels.

    Args:
        class_labels (List[str]): A list containing class labels.
    """

    def __init__(self, class_labels: List[str]):
        self.class_labels: List[str] = class_labels

    def __call__(self, indices: List[int]) -> List[str]:
        """
        Convert indices to class labels.

        Args:
            indices (List[int]): List containing indices.

        Returns:
            List[str]: List containing class labels corresponding to the indices.
        """
        return [self.class_labels[idx] for idx in indices]