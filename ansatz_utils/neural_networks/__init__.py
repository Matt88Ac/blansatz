from .activations import get_activation, nn
from .aggregations import get_aggregation
from .mlp import MLP
from .deepsets import DeepSets


def get_model(model_name: str, *args, **kwargs) -> nn.Module:
    if model_name.lower() == 'mlp':
        return MLP(*args, **kwargs)

    elif model_name.lower() in ['ds', 'deepsets', 'deepset']:
        return DeepSets(*args, **kwargs)




