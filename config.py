class BaseConfig:
    def __init__(self):
        pass


class ImageVAEConfig(BaseConfig):
    """Basic Image VAE model hparams."""
    def __init__(self, device='cpu'):
        self.device = device

    daisy_chain_variables = False
    batch_size = 64
    hidden_size = 32
    initializer = 'uniform_unit_scaling'
    initializer_gain = 1.0
    weight_decay = 0.0

    # VAE hparams
    base_depth = 32
    bottleneck_bits = 32

    # loss hparams
    kl_beta = 300 #300
    free_bits_div = 4
    free_bits = 0.15

    # data format hparams
    num_categories = 62

    # problem hparams (required, don't modify)
    absolute = False
    just_render = True
    plus_render = False
