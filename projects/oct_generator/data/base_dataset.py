"""
Minimal BaseDataset for pseudo-OCT cGAN training.

This version keeps:
- BaseDataset abstract class (repo-style dataset skeleton)
- get_transform(): Grayscale (optional) + ToTensor + Normalize

It removes:
- resize/crop/flip augmentations
- get_params and helper functions (__crop, __flip, __scale_width, __make_power_2, ...)
because images are already preprocessed (e.g., 256x256) offline.
"""

import torch.utils.data as data # pytorchun dataset/dataloader altyapısı burada
import torchvision.transforms as transforms # görüntüleri modele vermeden önce dönüştürmek için kullanılan dönüşümler (Totensor, normalize vs.)
from abc import ABC, abstractmethod # ABC= abstract base class 
# abstractmethod >>alt sınıfların mutlaka implement etmesi gereken fonksiyonları işaretler

# BaseDataset de kurallar koyulur
class BaseDataset(data.Dataset, ABC): # data.Dataset → PyTorch’un dataset formatına uyum için
    """Abstract base class (ABC) for datasets."""

    def __init__(self, opt):  # Dataset oluşturulurken çağrılır
        """
        Save experiment options.

        Parameters:
            opt: Options object. Must contain at least `dataroot` if you use it.
        """
        self.opt = opt # opt u class içinde saklar ki başka fonksiyonlar (mesela __getitem__) erişebilsin 
        self.root = opt.dataroot # dataroot >> datasetin ana klasör yolunu tutar (:\ML\pseudo_oct model data set)

    

    @abstractmethod
    def __len__(self):
        """Dataset kaç örnek içeriyor"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """ Index numaralı örneği getir"""
        raise NotImplementedError


def get_transform(opt=None, grayscale: bool = False):
    """
   transform pipelıneı oluşturuluyor ; görüntü--> tensor--> normalize edilmiş tensor

    """
    t = [] # transform işlem listesi oluşturuluyor bu liste ToTensor() ve Normalize() adım işlemlerini  sırayla kaydeder

    t.append(transforms.ToTensor()) 
    ''' PIL image ([0-255]) uint8 -> Pytorch tensor ([0-1]) float32
    Fundus --> [3, 256, 256]
    OCT --> [1, 256, 256]

    '''

    # Normalize to [-1, 1] for GANs (common with tanh output)
    if grayscale:
        t.append(transforms.Normalize((0.5,), (0.5,))) # tek kanal için --> transforms.Normalize(mean, std)
    else:
        t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))) # 3 kanal (R,G,B) için. her kanal için ayrı mean ve std kullanıyoruz

    return transforms.Compose(t)
