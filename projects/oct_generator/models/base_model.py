from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
import torch
from . import networks


class BaseModel(ABC):
    """
    Abstract base class for models.

    Every subclass should implement:
      - set_input()
      - forward()
      - optimize_parameters()
    """

    def __init__(self, opt):
        """
        Initialize common model settings.

        Parameters:
            opt: option object containing all experiment settings
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = opt.device
        self.save_dir = Path(opt.checkpoints_dir) / opt.name

        # Speed optimization when input image size is fixed
        if hasattr(opt, "preprocess") and opt.preprocess != "scale_width":
            torch.backends.cudnn.benchmark = True

        # These will be filled inside child model class (e.g. Pix2PixModel)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for plateau scheduler if needed

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add model-specific options if needed.
        Default: do nothing.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """
        Unpack input data from dataloader and apply preprocessing.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights.
        """
        pass

    def setup(self, opt):
        """
        Load networks if needed, print networks, and create schedulers.

        Called once after model creation.
        """
        # Load checkpoint if testing or continuing training
        if (not self.isTrain) or getattr(opt, "continue_train", False):
            load_suffix = f"iter_{opt.load_iter}" if getattr(opt, "load_iter", 0) > 0 else opt.epoch
            self.load_networks(load_suffix)

        # Print model summary
        self.print_networks(getattr(opt, "verbose", False))

        # Create schedulers only during training
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def eval(self):
        """
        Set all models to evaluation mode.
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """
        Run forward pass in test mode without gradient calculation.
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """
        Optional function for extra visualization outputs.
        Child classes can override this if needed.
        """
        pass

    def get_image_paths(self):
        """
        Return image paths for current batch.
        """
        return self.image_paths

    def update_learning_rate(self):
        """
        Update learning rate for all schedulers.
        Usually called at the end of each epoch.
        """
        old_lr = self.optimizers[0].param_groups[0]["lr"]

        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        new_lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"learning rate {old_lr:.7f} -> {new_lr:.7f}")

    def get_current_visuals(self):
        """
        Return current images for visualization/logging.
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """
        Return current losses as an ordered dictionary.
        Looks for attributes like self.loss_G_GAN, self.loss_D_fake, etc.
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))
        return errors_ret

    def save_networks(self, epoch):
        """
        Save all networks to disk.

        Example saved filenames:
            latest_net_G.pth
            latest_net_D.pth
            10_net_G.pth
            10_net_D.pth
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_net_{name}.pth"
                save_path = self.save_dir / save_filename
                net = getattr(self, "net" + name)
                torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch):
        """
        Load all networks from disk.
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = self.save_dir / load_filename
                net = getattr(self, "net" + name)

                print(f"loading the model from {load_path}")
                state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """
        Print total parameter count and optionally full network architecture.
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = sum(param.numel() for param in net.parameters())

                if verbose:
                    print(net)

                print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requires_grad for networks to avoid unnecessary gradient computation.

        Parameters:
            nets: a network or a list of networks
            requires_grad (bool): whether gradients should be enabled
        """
        if not isinstance(nets, list):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad