import torch

from datasets import weird_image_net, Normalizer
from model.library.base import ModelLibrary
from .base import TransformerModel
import clip
from torch import nn


class ClipWrapper(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.clip.encode_image(x)


class CLIP(TransformerModel):
    def __init__(self, o: int, batch_size: int):
        clip_options = ['RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        im_size = [288, 384, 224, 224, 224, 336]

        def get_clip(option: int = 0) -> torch.nn.Module:
            """ Hats off to: https://github.com/openai/CLIP """
            return ClipWrapper(clip.load(clip_options[option])[0])
        image_size = im_size[o]
        normalizer = Normalizer(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        super().__init__(f'CLIP{o}_{clip_options[o]}', get_clip, {'option': o}, image_size, batch_size, normalizer)


clip_models = ModelLibrary(other_models=[
    CLIP(0, 18),
    CLIP(1, 18),
    CLIP(2, 18),
    CLIP(3, 18),
    CLIP(4, 18),
    CLIP(5, 18),
])



#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

#        im_size = [224, 224, 288, 384, 224, 224]

# RN50: 224 		Image features shape: torch.Size([1, 1024])
# RN101: 224		Image features shape: torch.Size([1, 512])
# RN50x4: 288		Image features shape: torch.Size([1, 640])
# RN50x16: 384		Image features shape: torch.Size([1, 768])
# RN50x64: 448		Image features shape: torch.Size([1, 1024])
# ViT-B/32: 224		Image features shape: torch.Size([1, 512])
# ViT-B/16: 224		Image features shape: torch.Size([1, 512])
# ViT-L/14: 224		Image features shape: torch.Size([1, 768])
# ViT-L/14@336px: 336	Image features shape: torch.Size([1, 768])
