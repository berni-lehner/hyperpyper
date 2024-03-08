class ProjectTransform:
    """
    A data transformation class that applies a given data projector to transform input data to a lower dimension.

    Parameters:
    -----------
    projector : object
        A dimensionality reduction model or function that implements a 'fit_transform' method to project input data.
        Compatible dimensionality reducers include UMAP, t-SNE, Random Projections, and others following scikit-learn's API.

    Methods:
    --------
    __call__(X)
        Apply the data projection to input data X.

    Attributes:
    -----------
    projector : object
        The dimensionality reduction model used for transforming data.

    Notes:
    ------
    Some dimensionality reducers like UMAP and t-SNE require training before being used with ProjectTransform.
    They must be trained using their 'fit' or 'fit_transform' methods on the entire dataset, as they cannot be trained with mini-batches.
    """
    def __init__(self, projector):
        self.projector = projector

    def __call__(self, X):
        X_trans = self.projector.transform(X)
        
        return X_trans
    
    def __repr__(self):
        return f"{self.__class__.__name__}(self.projector={self.projector})"