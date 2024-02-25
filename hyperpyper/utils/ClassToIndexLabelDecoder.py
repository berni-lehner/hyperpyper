from typing import List, Dict

class ClassToIndexLabelDecoder:
    """
    Decoder class to convert class labels to indices.

    Args:
        class_labels (List[str]): A list containing class labels.

    Attributes:
        class_labels (List[str]): A list containing class labels.
        label_to_index_map (Dict[str, int]): A dictionary mapping class labels to indices.
    """

    def __init__(self, class_labels: List[str]):
        self.class_labels: List[str] = class_labels
        self.label_to_index_map: Dict[str, int] = {label: idx for idx, label in enumerate(class_labels)}

    def __call__(self, labels: List[str]) -> List[int]:
        """
        Convert class labels to indices.

        Args:
            labels (List[str]): List containing class labels.

        Returns:
            List[int]: List containing indices corresponding to the class labels.
        """
        return [self.label_to_index_map[label] for label in labels]