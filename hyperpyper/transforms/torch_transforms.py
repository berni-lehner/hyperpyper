import torch
import torch.nn as nn


class TensorToNumpy(object):
    def __call__(self, X):
        return X.numpy()
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class FlattenTensor:
    def __call__(self, tensor):
        return tensor.view(-1)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class SqueezeTensor:
    def __call__(self, tensor):
        return tensor.squeeze()

    def __repr__(self):
        return self.__class__.__name__ + '()'
            

class UnsqueezeTensor:
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(self.dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"

    
class ToDevice:
    def __init__(self, device=None):
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        
    def __call__(self, data):
        return data.to(self.device)

    
    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"


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
    
    #TODO: proper string construction with properties
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



# TODO: test!!!
class PyTorchExplain:
    """
    Computes Integrated Gradients using a PyTorch model.

    Methods:
    __call__(input_image, target_class=None): Computes Integrated Gradients for the input image with respect to the target class.
    """

    def __init__(self, model, baseline=None, steps=100, device=None):
        """
        Initialize the PyTorchExplain.

        Args:
        model (torch.nn.Module): The PyTorch model for which Integrated Gradients will be computed.
        baseline (torch.Tensor or None, optional): The baseline input for the integration path. If None, a zero baseline is used. Default: None.
        steps (int, optional): The number of steps for the numerical integration. Default: 100.
        device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
        """
        self.model = model
        self.baseline = baseline
        self.steps = steps

        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Set the module in evaluation mode
        self.model.eval()

        if self.baseline is not None:
            self.baseline = self.baseline.to(self.device)

    def __call__(self, input_image, target_class=None):
        """
        Computes Integrated Gradients for the input image with respect to the target class.

        Args:
        input_image (torch.Tensor): The input image tensor.
        target_class (int or None, optional): The target class index. If None, the class with the highest probability is used. Default: None.

        Returns:
        torch.Tensor: The computed Integrated Gradients.
        """
        input_image = input_image.to(self.device)
        input_image.requires_grad_()

        if target_class is None:
            output = self.model(input_image)
            target_class = output.argmax()

        baseline = self.baseline or torch.zeros_like(input_image)
        scaled_inputs = [baseline + (float(i) / self.steps) * (input_image - baseline) for i in range(self.steps + 1)]

        integrated_gradients = torch.zeros_like(input_image)
        for scaled_input in scaled_inputs:
            gradients = self._compute_gradients(scaled_input, target_class)
            integrated_gradients += gradients

        integrated_gradients *= (input_image - baseline) / self.steps

        return integrated_gradients

    def _compute_gradients(self, input_image, target_class):
        """
        Computes gradients of the target class output with respect to the input image.

        Args:
        input_image (torch.Tensor): The input image tensor.
        target_class (int): The target class index.

        Returns:
        torch.Tensor: The computed gradients.
        """
        self.model.zero_grad()
        output = self.model(input_image)
        output[0, target_class].backward()

        return input_image.grad.clone()
