import numpy as np
from sklearn.base import BaseEstimator


class RandomProjection(BaseEstimator):
    """
    A class for performing random projection for dimensionality reduction.

    Parameters:
    -----------
    output_dim : int
        The desired output dimension after random projection.

    Methods:
    --------
    transform(X)
        Apply random projection to input data X.

    Attributes:
    -----------
    output_dim : int
        The desired output dimension after random projection.
    random_matrix : numpy.ndarray or None
        The random projection matrix used for the transformation.

    Notes:
    ------
    RandomProjection is a technique for reducing the dimensionality of data by projecting it onto a random matrix.
    The random projection matrix is generated when the first transformation is applied.
    
    This class supports both 1-D and 2-D input data. If the input is 1-D, it will be reshaped to 2-D (1 sample).
    If the input is already 2-D, the transformation will be applied row-wise.

    Examples:
    ---------
    # Creating a RandomProjection instance
    rp = RandomProjection(output_dim=10)
    
    # Applying random projection to input data
    X = np.random.randn(100, 20)  # Replace with your data
    X_transformed = rp.transform(X)
    """
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.random_matrix = None

    def fit_transform(self, X):
        """
        Alias for transform().
        """        
        return self.transform(X)
    
    def transform(self, X):
        if self.random_matrix is None:
            input_dim = X.shape[0]
            self.random_matrix = np.random.randn(self.output_dim, input_dim)

        return self._transform_array(X)

    def _transform_array(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim > 2:
            raise ValueError("Input dimension should be 1-D or 2-D.")
            
        return np.dot(X, self.random_matrix.T)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
