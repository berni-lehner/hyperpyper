import ssl
import inspect
from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from ..utils import DataSetDumper

class VisionDatasetDumper():
    """
    Download and prepare a torchvision dataset by extracting its images into subfolders based on their labels.


    Args:
        dataset (torchvision.datasets.VisionDataset): The dataset object.
        root (pathlib.Path): Root directory where the dataset is located or will be downloaded.
        dst (pathlib.Path): Destination directory for the prepared dataset.
        train (bool, optional): Whether the dataset is for training. Defaults to True.
        download (bool, optional): Whether to download the dataset if not found locally. Defaults to True.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.    
    """
    def __init__(self,
                dataset: VisionDataset,
                root: Path,
                dst: Path,
                train: bool=True,
                split=None,
                download: bool=True,
                transform=ToTensor(),
                target_transform=None):
        # Sanity check if dataset is really from torchvision.datasets
        if not issubclass(dataset, VisionDataset):
            raise ValueError("The dataset must be from torchvision.datasets (e.g. torchvision.datasets.CIFAR10).")

        self.dataset = dataset
        self.root = root
        self.dst = dst
        self.train = train
        self.split = split
        self.download = download
        self.transform = transform
        self.target_transform = target_transform


        # this prevents the following error when trying to download the dataset:
        # SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1007)
        ssl._create_default_https_context = ssl._create_unverified_context

        # call the constructor with the appropriate parameters (train AND/OR split)
        params = inspect.signature(dataset.__init__).parameters
        if 'train' in params:
            if 'split' in params:
                self.ds = dataset(root=self.root, train=self.train, split=self.split,
                    transform=self.transform, target_transform=self.target_transform,
                    download=self.download)
            else:
                self.ds = dataset(root=self.root, train=self.train,
                    transform=self.transform, target_transform=self.target_transform,
                    download=self.download)

        else:
            if self.split is None:
                self.split = 'train' if self.train else 'test'
            self.ds = dataset(root=self.root, split=self.split, transform=self.transform,
                target_transform=self.target_transform, download=self.download)

        


    def dump(self) -> VisionDataset:
        """
        Extract images into subfolders based on their labels.

        Returns:
            torch.utils.data.Dataset: Prepared dataset.
        """
        if not self.dst.exists():
            DataSetDumper(self.ds, self.dst).dump()

        return self.ds