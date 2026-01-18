from torch import nn
from .activations import get_activation
from .aggregations import get_aggregation
from .dropout import DropoutEquivariant
from .mlp import MLP
from .deepsets import DeepSets
from .attention import Transformer
from .pivot_mlp import PivotMLP


def get_model(model_name: str, *args, **kwargs) -> nn.Module:
    if model_name.lower() == 'mlp':
        return MLP(*args, **kwargs)

    elif model_name.lower() in ['ds', 'deepsets', 'deepset']:
        return DeepSets(*args, **kwargs)

    elif model_name.lower() in ['attention', 'transformer']:
        return Transformer(*args, **kwargs)

    elif model_name.lower() in ['pivot', 'pmlp', 'pivot_mlp']:
        return PivotMLP(*args, **kwargs)

    
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Available models are: 'mlp', 'deepsets', 'attention', 'pivot'.")




