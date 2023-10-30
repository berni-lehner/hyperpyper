import numpy as np
import hashlib
import torch
from sklearn.pipeline import Pipeline
from pathlib import Path

from ..utils import Pickler


class PipelineCache:
    """
    A class that serves as a wrapper around any sklearn pipeline to cache the results.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): The sklearn pipeline to be wrapped.
        cache_path (str or pathlib.Path): The directory path where the cache files will be stored.
    """

    def __init__(self, pipeline, cache_path):
        self.pipeline = pipeline
        self.cache_path = Path(cache_path)


    def _get_hash(self, data):
        """
        Compute the MD5 hash value of the input data.

        Parameters:
            data: The data to be hashed.

        Returns:
            str: The computed hash value.
        """
        # Compute the hash of the binary representation
        if isinstance(data, np.ndarray):
            hash_value = hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, torch.Tensor):
            hash_value = hashlib.sha256(data.cpu().detach().numpy().tobytes()).hexdigest()
        else:
            raise ValueError("Invalid input type. Expected np.ndarray or torch.Tensor.")
            
        return hash_value

        #data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
        #return data_hash


    def _get_cache_file(self, hash_value, prefix=''):
        """
        Generate the cache file path based on the hash value.

        Parameters:
            hash_value (str): The hash value used to form the cache file name.

        Returns:
            Path: The full cache file path.
        """
        filename = f"{hash_value}.pkl"
        return self.cache_path / prefix / filename


    def fit(self, X, y=None):
        """
        Fit the pipeline and cache the results if necessary.

        Parameters:
            X: The input data for fitting the pipeline.
            y: The target values for fitting the pipeline. Default is None.

        Returns:
            The fitted pipeline or the cached results.
        """
        hash_value = self._get_hash(X)
        cache_file = self._get_cache_file(hash_value, prefix='fit')

        print(cache_file)

        if cache_file is None:
            results = self.pipeline.fit(X, y)

        elif Path(cache_file).exists():
            results = Pickler.load_data(cache_file)
        else:
            results = self.pipeline.fit(X, y)
            Pickler.save_data(results, cache_file)
        return results


    def transform(self, X):
        """
        Transform the input data using the pipeline and cache the results if necessary.

        Parameters:
            X: The input data for transforming.

        Returns:
            The transformed data or the cached results.
        """
        hash_value = self._get_hash(X)
        cache_file = self._get_cache_file(hash_value, prefix='transform')

        if cache_file is None:
            results = self.pipeline.transform(X)

        elif Path(cache_file).exists():
            results = Pickler.load_data(cache_file)
        else:
            results = self.pipeline.transform(X)
            Pickler.save_data(results, cache_file)
        return results


    def fit_transform(self, X):
        """
        Fits the pipeline and transforms the input data using the pipeline and cache the results if necessary.

        Parameters:
            X: The input data for fitting the pipeline and transforming.

        Returns:
            The transformed data or the cached results.
        """
        hash_value = self._get_hash(X)
        cache_file = self._get_cache_file(hash_value, prefix='fit_transform')

        if cache_file is None:
            results = self.pipeline.fit_transform(X)

        elif Path(cache_file).exists():
            results = Pickler.load_data(cache_file)
        else:
            results = self.pipeline.fit_transform(X)
            Pickler.save_data(results, cache_file)
        return results


    def predict(self, X):
        """
        Make predictions on the input data using the pipeline and cache the results if necessary.

        Parameters:
            X: The input data for making predictions.

        Returns:
            The predicted values or the cached results.
        """
        hash_value = self._get_hash(X)
        cache_file = self._get_cache_file(hash_value, prefix='predict')

        if cache_file is None:
            results = self.pipeline.predict(X)

        elif Path(cache_file).exists():
            results = Pickler.load_data(cache_file)
        else:
            results = self.pipeline.predict(X)
            Pickler.save_data(results, cache_file)
        return results


    def predict_proba(self, X):
        """
        Compute class probabilities on the input data using the pipeline and cache the results if necessary.

        Parameters:
            X: The input data for computing class probabilities.

        Returns:
            The class probabilities or the cached results.
        """
        hash_value = self._get_hash(X)
        cache_file = self._get_cache_file(hash_value, prefix='predict_proba')

        if cache_file is None:
            results = self.pipeline.predict_proba(X)

        elif Path(cache_file).exists():
            results = Pickler.load_data(cache_file)
        else:
            results = self.pipeline.predict_proba(X)
            Pickler.save_data(results, cache_file)
        return results
