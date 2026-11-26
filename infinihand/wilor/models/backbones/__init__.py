from .vit import vit


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE != 'vit':
        raise ValueError('InfiniHand uses the WiLoR ViT backbone')
    return vit(cfg)
