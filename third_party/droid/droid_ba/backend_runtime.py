"""DROID feature extraction and global BA without an online SLAM frontend."""
from collections import OrderedDict

import torch

from depth_video import DepthVideo
from droid_backend import DroidBackend
from droid_net import DroidNet


class FeatureEncoder:
    def __init__(self, net, device='cuda:0'):
        self.cnet, self.fnet = net.cnet, net.fnet
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.amp.autocast('cuda', enabled=True)
    def context_encoder(self, image):
        net, inp = self.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda', enabled=True)
    def feature_encoder(self, image):
        return self.fnet(image).squeeze(0)


class DroidBackendRuntime:
    def __init__(self, args):
        self.net = DroidNet()
        state = OrderedDict((key.removeprefix('module.'), value) for key, value in
                            torch.load(args.weights, map_location='cpu', weights_only=True).items())
        for key in ('update.weight.2.weight', 'update.weight.2.bias',
                    'update.delta.2.weight', 'update.delta.2.bias'):
            state[key] = state[key][:2]
        self.net.load_state_dict(state, strict=True)
        self.net.to('cuda:0').eval().requires_grad_(False)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)
        self.encoder = FeatureEncoder(self.net)
        self.backend = DroidBackend(self.net, self.video, args)
