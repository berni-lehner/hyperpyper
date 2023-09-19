import pickle
from pathlib import Path

class Pickler:
    """
    A utility class for saving and loading data using pickle serialization.

    Methods:
    - save_data(data, filename): Save data to a binary file using pickle serialization.
    - load_data(filename): Load data from a binary file using pickle deserialization.
    """
    @staticmethod
    def save_data(data, filename):
        """
        Save data to a binary file using pickle serialization.

        Parameters:
        data (Any): The data to be saved.
        filename (str): The name of the file to save the data to.

        Returns:
        None
        """
        resultpath = Path(filename).parent
        if not resultpath.exists():
            resultpath.mkdir(parents=True)

        with open(filename, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    @staticmethod
    def load_data(filename):
        """
        Load data from a binary file using pickle deserialization.

        Parameters:
        filename (str): The name of the file to load data from.

        Returns:
        Any: The loaded data.
        """
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

        return data