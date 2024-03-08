from pathlib import Path
from collections import Counter
import torchvision
from torchvision.transforms import ToTensor
from torch import Tensor


#from ..transforms import DummyPIL
#from ..aggregator import DataSetAggregator


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
            # Get target list
            if hasattr(self.dataset, 'targets'):
                # Determine unique targets
                if isinstance(self.dataset.targets, Tensor):
                    all_targets = self.dataset.targets.numpy()
                else:
                    all_targets = self.dataset.targets
                ctr = Counter(all_targets)
                targets = ctr.keys()
                print(targets)
            elif hasattr(self.dataset, 'labels'):
                # Determine unique targets
                if isinstance(self.dataset.labels, Tensor):
                    all_targets = self.dataset.labels.numpy()
                else:
                    all_targets = self.dataset.labels
                ctr = Counter(all_targets)
                targets = ctr.keys()
                print(targets)
            else: # Iterate the dataset and figure out the targets
                raise ValueError("Invalid input type. DataSet needs to be from torchvision with attribute 'targets'.")
                assert False, "Invalid input type. DataSet needs to be from torchvision with attribute 'targets'."
                # TODO: test and activate
                org_transform = self.dataset.transform
                # Use a transform pipeline with dummy data to quickly load the targets
                dummy_transform = Compose(
                    [
                        DummyPIL(),
                        ToTensor(),
                    ]
                )
                self.dataset.transform = dummy_transform
                agg = DataSetAggregator(self.dataset, batch_size=1)
                dummy_result = agg.transform()

                # Determine unique targets
                ctr = Counter(dummy_result[1].numpy())
                targets = ctr.keys()
                print(targets)

                # Set the original transform again
                self.dataset.transform = org_transform

        # Create subfolders for each class label
        for t in targets:
            target_dir = Path(self.root, str(t))
            target_dir.mkdir(parents=True, exist_ok=True)

        # Organize the dataset into subfolders
        for idx, (image, target) in enumerate(self.dataset):
            target_dir = Path(self.root, str(target))
            image_path = Path(target_dir, f"{idx}.png")
            torchvision.utils.save_image(image, image_path)