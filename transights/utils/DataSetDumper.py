from pathlib import Path
from collections import Counter
import torchvision
from torchvision.transforms import ToTensor


from ..transforms import DummyPIL
from ..aggregator import DataSetAggregator


class DataSetDumper():
    """
    A class for dumping images in a data set into folders that corresond to their target.

    Parameters:
    -----------
    dataset : DataSet
        The Dataset.
    root : pathlib.Path
        The result root path.

    """
    def __init__(self, dataset, root):
        self.dataset = dataset
        self.root = root




    def dump(self, targets=[]):
        class_labels = []

        # Check if targets are given
        if not len(targets):
            if not hasattr(self.dataset, 'targets'):
                raise ValueError("Invalid input type. DataSet needs to be from torchvision with attribute 'targets'.")
            # Determine unique targets
            ctr = Counter(self.dataset.targets)
            targets = ctr.keys()
            print(targets)
        # Create subfolders for each class label
        for t in targets:
            target_dir = Path(self.root, str(t))
            target_dir.mkdir(parents=True, exist_ok=True)

        # Organize the dataset into subfolders
        for idx, (image, target) in enumerate(self.dataset):
            target_dir = Path(self.root, str(target))
            image_path = Path(target_dir, f"{idx}.png")
            torchvision.utils.save_image(image, image_path)