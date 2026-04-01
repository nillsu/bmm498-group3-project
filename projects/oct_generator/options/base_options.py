import argparse
from pathlib import Path



class BaseOptions:
    
    def __init__(self):
        self.initialized = False
        self.isTrain = True  # train/test flag 

    def initialize(self, parser):

       
        # BASIC

        parser.add_argument("--dataroot", required=True, help="dataset root folder")
        parser.add_argument("--name", type=str, default="pseudo_oct_experiment")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")

        # IMPORTANT (dataset uses this!)
        parser.add_argument("--phase", type=str, default="train")  # FIXED

      
        # MODEL (FIXED)
       
        parser.add_argument("--model", type=str, default="pix2pix")

        parser.add_argument("--input_nc", type=int, default=3)   # fundus (RGB)
        parser.add_argument("--output_nc", type=int, default=1)  # OCT (grayscale)

        parser.add_argument("--ngf", type=int, default=64)
        parser.add_argument("--ndf", type=int, default=64)

        parser.add_argument("--netG", type=str, default="unet_256")
        parser.add_argument("--netD", type=str, default="basic")

        parser.add_argument("--n_layers_D", type=int, default=3)
        parser.add_argument("--norm", type=str, default="batch")

        parser.add_argument("--init_type", type=str, default="normal")
        parser.add_argument("--init_gain", type=float, default=0.02)

        parser.add_argument("--no_dropout", action="store_true")

        parser.add_argument("--gan_mode", type=str, default="vanilla")

        
        # DATASET (FIXED)
      
        parser.add_argument("--dataset_mode", type=str, default="aligned")
        parser.add_argument("--direction", type=str, default="AtoB")

        parser.add_argument("--serial_batches", action="store_true")
        parser.add_argument("--num_threads", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=1)

        parser.add_argument("--max_dataset_size", type=int, default=float("inf"))

        parser.add_argument("--dirA", type=str, default="fundus_pse-model")
        parser.add_argument("--dirB", type=str, default="oct_pse-model")

        # IMPORTANT (resize YOK)
        parser.add_argument("--load_size", type=int, default=256)
        parser.add_argument("--crop_size", type=int, default=256)
        parser.add_argument("--preprocess", type=str, default="none")
        parser.add_argument("--no_flip", action="store_true")

        
        # TRAINING LOSSES (ADDED)
        
        parser.add_argument("--lambda_L1", type=float, default=100.0)
        parser.add_argument("--lambda_adv", type=float, default=1.0)


        
        # OPTIMIZER (normalde train_options.py içerisindeydi)
       
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--beta1", type=float, default=0.5)

        
        # TRAINING SCHEDULE 
        
        parser.add_argument("--n_epochs", type=int, default=100)
        parser.add_argument("--n_epochs_decay", type=int, default=100)

        parser.add_argument("--lr_policy", type=str, default="linear")
        parser.add_argument("--epoch_count", type=int, default=1)
        parser.add_argument("--lr_decay_iters", type=int, default=50)

        
        # LOADING / DEBUG
        
        parser.add_argument("--epoch", type=str, default="latest")
        parser.add_argument("--load_iter", type=int, default=0)

        parser.add_argument("--verbose", action="store_true")

        self.initialized = True
        return parser

    def gather_options(self): # tüm argümanları alıp opt objesine çeviriyor

        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt): # tüm parametreleri ekrana yazdırır dosyaya kaydeder (opt.txt)

        message = ""
        message += "----------------- Options ---------------\n"

        for k, v in sorted(vars(opt).items()):
            message += f"{k:>25}: {str(v):<30}\n"

        message += "----------------- End -------------------"
        print(message)

        # save to disk
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        expr_dir.mkdir(parents=True, exist_ok=True) # util kaldırıldı

        file_name = expr_dir / "opt.txt"
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self): # hepsini birleştirir

        opt = self.gather_options()
        opt.isTrain = self.isTrain  

        self.print_options(opt)

        self.opt = opt
        return self.opt