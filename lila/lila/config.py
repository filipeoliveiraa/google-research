# coding=utf-8

class LILAConfig:

    _configs = {
        'vits': {'features': 2*128, 'out_channels': [48, 96, 192, 384],       'intermediate_layer_idx': [2, 5, 8, 11]},
        'vitb': {'features': 2*192, 'out_channels': [96, 192, 384, 768],      'intermediate_layer_idx': [2, 5, 8, 11]},
        'vitl': {'features': 2*256, 'out_channels': [256, 512, 1024, 1024],   'intermediate_layer_idx': [4, 11, 17, 23]},
        'vitg': {'features': 2*384, 'out_channels': [1536, 1536, 1536, 1536], 'intermediate_layer_idx': [9, 19, 29, 39]}
    }

    def __getitem__(self, encoder_name):
        base_config = dict(self._configs[encoder_name[-6:-2]])
        base_config["patch_size"] = int(encoder_name[-2:])
        return base_config

lila_config = LILAConfig()
