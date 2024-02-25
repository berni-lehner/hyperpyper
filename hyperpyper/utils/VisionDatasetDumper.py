import ssl
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
    """
    def __init__(self, dataset: VisionDataset, root: Path, dst: Path, train: bool=True, download: bool=True):
        # Sanity check if dataset is really from torchvision.datasets
        if not issubclass(dataset, VisionDataset):
            raise ValueError("The dataset must be from torchvision.datasets (e.g. torchvision.datasets.CIFAR10).")

        self.dataset = dataset
        self.root = root
        self.dst = dst
        self.train = train
        self.download = download

        # this prevents the following error when trying to download the dataset:
        # SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1007)
        ssl._create_default_https_context = ssl._create_unverified_context

        self.ds = dataset(root=self.root, train=self.train, transform=ToTensor(), download=self.download)
        


    def dump(self) -> VisionDataset:
        """
        Extract images into subfolders based on their labels.

        Returns:
            torch.utils.data.Dataset: Prepared dataset.
        """
        if not self.dst.exists():
            DataSetDumper(self.ds, self.dst).dump()

        return self.ds