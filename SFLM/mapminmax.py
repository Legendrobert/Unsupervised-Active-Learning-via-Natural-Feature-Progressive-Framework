import torch

class MapMinMax:
    """
    PyTorch implementation of MATLAB's mapminmax function
    """
    def __init__(self, y_min=-1.0, y_max=1.0):
        """
        Initialize the mapminmax parameters.
        
        Args:
            y_min (float): Lower bound of normalization (default -1.0)
            y_max (float): Upper bound of normalization (default 1.0)
        """
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = None
        self.x_max = None
        self.x_range = None
        
    def fit(self, x):
        """
        Calculate normalization parameters from input data.
        
        Args:
            x (torch.Tensor): Input data tensor
            
        Returns:
            self: Returns the instance itself
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Store min and max values along each dimension (feature)
        self.x_min = torch.min(x, dim=1, keepdim=True)[0]
        self.x_max = torch.max(x, dim=1, keepdim=True)[0]
        self.x_range = self.x_max - self.x_min
        
        # Handle constant features (avoid division by zero)
        self.x_range[self.x_range == 0] = 1.0
        
        return self
    
    def transform(self, x):
        """
        Apply normalization using calculated parameters.
        
        Args:
            x (torch.Tensor): Input data tensor to normalize
            
        Returns:
            torch.Tensor: Normalized data
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Apply linear scaling
        y = (x - self.x_min) / self.x_range * (self.y_max - self.y_min) + self.y_min
        return y
    
    def fit_transform(self, x):
        """
        Fit to data and transform it in one step.
        
        Args:
            x (torch.Tensor): Input data tensor
            
        Returns:
            torch.Tensor: Normalized data
        """
        return self.fit(x).transform(x)
    
    def inverse_transform(self, y):
        """
        Transform data back to original scale.
        
        Args:
            y (torch.Tensor): Normalized data tensor
            
        Returns:
            torch.Tensor: Data in original scale
        """
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
            
        # Apply inverse scaling
        x = (y - self.y_min) / (self.y_max - self.y_min) * self.x_range + self.x_min
        return x
    
    def get_params(self):
        """
        Get the parameters for later use (similar to MATLAB's 'PS' structure).
        
        Returns:
            dict: Dictionary containing the normalization parameters
        """
        return {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'x_range': self.x_range,
            'y_min': self.y_min,
            'y_max': self.y_max
        }
    
    def set_params(self, params):
        """
        Set parameters from a previously saved state.
        
        Args:
            params (dict): Dictionary containing the normalization parameters
            
        Returns:
            self: Returns the instance itself
        """
        self.x_min = params['x_min']
        self.x_max = params['x_max']
        self.x_range = params['x_range']
        self.y_min = params['y_min']
        self.y_max = params['y_max']
        return self

    def save_params(self, filename):
        """
        Save parameters to a file.
        
        Args:
            filename (str): Path to save the parameters
        """
        torch.save(self.get_params(), filename)
    
    def load_params(self, filename):
        """
        Load parameters from a file.
        
        Args:
            filename (str): Path to the saved parameters file
            
        Returns:
            self: Returns the instance itself
        """
        params = torch.load(filename)
        return self.set_params(params)