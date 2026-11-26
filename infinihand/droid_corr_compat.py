"""Inference-only adapter for DROID's indexed (6-argument) alt-correlation ABI.

Bundled DROID uses the older 4-argument ABI. Current DROID CUDA builds perform
feature indexing and /4 normalization internally. Do not pre-divide features.
No vendor/training file is changed; patch only this backend process's graph class.
"""
def install_corr_compat():
    import torch
    import torch.nn.functional as F
    import droid_backends
    import factor_graph
    signature = droid_backends.altcorr_forward.__doc__ or ''
    if 'arg5' not in signature:
        return 'legacy_4arg'

    class IndexedAltCorrBlock:
        def __init__(self, fmaps, num_levels=4, radius=3):
            self.radius = radius
            B,N,C,H,W = fmaps.shape
            self.pyramid = []
            x = fmaps.reshape(B*N,C,H,W).float()
            for _ in range(num_levels):
                self.pyramid.append(x.view(B,N,C,x.shape[-2],x.shape[-1]).contiguous())
                x = F.avg_pool2d(x,2,stride=2)

        def __call__(self, coords, ii, jj):
            if torch.is_grad_enabled():
                raise RuntimeError('Indexed DROID adapter is inference-only')
            coords = coords.permute(0,1,4,2,3).float().contiguous()
            result = []
            for level, fmap in enumerate(self.pyramid):
                corr, = droid_backends.altcorr_forward(self.pyramid[0],fmap,
                    (coords/(2**level)).contiguous(),ii.long().contiguous(),jj.long().contiguous(),self.radius)
                result.append(corr.flatten(2,3))
            return torch.stack(result,dim=2).flatten(2,3).contiguous()

    factor_graph.AltCorrBlock = IndexedAltCorrBlock
    return 'indexed_6arg_fp32'
