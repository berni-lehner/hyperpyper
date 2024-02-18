import torch
import torch.nn as nn

class PyTorchEmbedding:
    """
    A utility class for extracting embeddings from a PyTorch model.

    Args:
    model (torch.nn.Module): The PyTorch model from which embeddings will be extracted.
    device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
    from_layer (int, optional): The index of the starting layer for slicing the model. Default: None.
    to_layer (int, optional): The index of the ending layer for slicing the model. Default: None.

    Methods:
    __slice_model(model, from_layer, to_layer): Slices the model to retain layers from `from_layer` to `to_layer`.
    __auto_slice(model): Automatically removes all linear layers from the end of the model.
    __call__(img): Extracts embeddings from the input image using the configured model.
    """

    def __init__(self, model, device=None, from_layer=None, to_layer=None):
        """
        Initialize the PyTorchEmbedding.

        Args:
        model (torch.nn.Module): The PyTorch model from which embeddings will be extracted.
        device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
        from_layer (int, optional): The index of the starting layer for slicing the model. Default: None.
        to_layer (int, optional): The index of the ending layer for slicing the model. Default: None.
        """
        if from_layer or to_layer:
            self.model = self.__slice_model(model, from_layer, to_layer)
        else:
            self.model = self.__auto_slice(model)

        # Set the module in evaluation mode
        self.model.eval()

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def __slice_model(self, model, from_layer=None, to_layer=None):
        """
        Slices the model to retain layers from `from_layer` to `to_layer`.

        Args:
        model (torch.nn.Module): The PyTorch model to be sliced.
        from_layer (int, optional): The index of the starting layer for slicing. Default: None.
        to_layer (int, optional): The index of the ending layer for slicing. Default: None.

        Returns:
        torch.nn.Module: The sliced model.
        """
        # Make model iterable
        mdl = nn.Sequential(*list(model.children()))

        return mdl[from_layer:to_layer]

    def __auto_slice(self, model):
        """
        Automatically removes all linear layers from the end of the model.

        Args:
        model (torch.nn.Module): The PyTorch model to be sliced.

        Returns:
        torch.nn.Module: The model with linear layers removed from the end.
        """
        # Make model iterable
        mdl = nn.Sequential(*list(model.children()))

        last_linear_layer_idx = None

        # Figure out how many linear layers are at the end
        for i, layer in reversed(list(enumerate(mdl))):
            if isinstance(layer, torch.nn.modules.linear.Linear):
                last_linear_layer_idx = i
            else:
                break

        return mdl[:last_linear_layer_idx]

    def __call__(self, img):
        """
        Extract embeddings from the input image using the configured model.

        Args:
        img (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The extracted embeddings.
        """
        if img.dim() != 4:
            # Add a batch dimension to the image tensor
            img = img.unsqueeze(0)

        # Pass the image through the model and get the embeddings
        with torch.no_grad():
            embeddings = self.model(img).detach()  # TODO: Not sure if we even need detach() here

        return embeddings
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class PyTorchOutput:
    """
    A utility class for obtaining model outputs from a PyTorch model.

    Args:
    model (torch.nn.Module): The PyTorch model to obtain outputs from.
    device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.

    Methods:
    __call__(img): Passes an image through the model and returns the resulting output.
    """

    def __init__(self, model, device=None):
        """
        Initialize the PyTorchOutput.

        Args:
        model (torch.nn.Module): The PyTorch model to obtain outputs from.
        device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
        """
        self.model = model

        # Set the module in evaluation mode
        self.model.eval()

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def __call__(self, img):
        """
        Passes an image through the model and returns the resulting output.

        Args:
        img (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The output tensor produced by the model.
        """
        if img.dim() != 4:
            # Add a batch dimension to the image tensor
            img = img.unsqueeze(0)

        # Pass the image through the model and get the output
        with torch.no_grad():
            result = self.model(img).detach()  # TODO: Not sure if we even need detach() here

        return result


    
    def __repr__(self):
        return self.__class__.__name__ + '()'