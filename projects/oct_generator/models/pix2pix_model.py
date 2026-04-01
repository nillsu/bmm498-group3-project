import torch  # Tensor işlemleri, loss hesapları, GPU’ya gönderme (.to(device)), optimizer
from .base_model import BaseModel
from . import networks # Generator/Discriminator’ı kuran fonksiyonlar burada


class Pix2PixModel(BaseModel): # Pix2PixModel, BaseModel’den miras alıyor --> Yani BaseModel’in fonksiyonları ve altyapısı Pix2PixModel’de de var
    """
    Pix2Pix model for paired image-to-image translation.

    Your project setting:
      - A: Fundus (RGB, 3-channel)
      - B: OCT B-scan (Grayscale, 1-channel)
      - Total loss:
          L_total = lambda_adv * L_adv + lambda_L1 * L1
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True): # train scriptde; parser = argparse.ArgumentParser() tanımlanır
        """
        Add/override options for your fundus->OCT pix2pix training.
        """
        # Keep aligned paired dataset, UNet generator defaults
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        '''
        BatchNorm kullan
        generator olarak UNet-256 seç
        paired dataset kullan
        '''

        if is_train: 
            # Pix2Pix uses no image pool buffer
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            '''
            vanilla--> klasik GAN loss 
            '''

            # Loss weights (your formulation)
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 (MAE) loss")
            parser.add_argument("--lambda_adv", type=float, default=1.0, help="weight for adversarial (GAN) loss")

        return parser

    def __init__(self, opt): 
        """
        Initialize the model.
        """
        BaseModel.__init__(self, opt)

        # Loss names for logging
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]

        # Visuals for saving / displaying
        self.visual_names = ["real_A", "fake_B", "real_B"] # Bu da eğitim sırasında görselleri kaydetmek/ekranda göstermek için

        # Networks to save/load
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        # Device
        self.device = opt.device

        # (Optional) safety checks for your project (fundus=3, oct=1)
        # Not hard requirements, but helps catch mismatched option configs early.
        if hasattr(opt, "input_nc") and opt.input_nc != 3:
            print(f"[WARN] opt.input_nc={opt.input_nc}. For fundus RGB you usually want input_nc=3.")
        if hasattr(opt, "output_nc") and opt.output_nc != 1:
            print(f"[WARN] opt.output_nc={opt.output_nc}. For OCT grayscale you usually want output_nc=1.")

        # Define Generator: G(A)->B
        self.netG = networks.define_G(  # modele ait bir Generator oluşturup adını netG koyduk 
            opt.input_nc,
            opt.output_nc,
            opt.ngf, # number of generator filters, default=64 --> artarak gider (*2)
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain, # burada oluşturulan değerler define_G içerisine gönderilir
        )

        # Define Discriminator: D([A,B]) where channels = input_nc + output_nc
        if self.isTrain:
            self.netD = networks.define_D( 
                opt.input_nc + opt.output_nc, #  D’ye A ile B (gerçek OCT B veya sahte OCT G(A))’yi birleştirip veriyoruz
                opt.ndf, # num of disc filters, default=64 
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
            )

        if self.isTrain:
            # GAN için kullanılacak kayıp fonksiyonunu oluşturup doğru cihaza taşıma işlemi (to(self.device)) 
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device) # GAN loss (gerçekçilik için)
            # L1Loss zaten Pytorch da torch.nn modülünde hazır halde bulunur
            self.criterionL1 = torch.nn.L1Loss() # L1 loss (piksel uyumu için)

            # Optimizers --> generator içindeki tüm öğrenilebilir ağırlıkları optimizera verilmesi 
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # tüm optimizerlar optimizers listesine konuyor
            self.optimizers.append(self.optimizer_G) 
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input): # aligned_dataset.py da __getitem__ altında oluşturulan dictionary kullanılır = input
        """
        Unpack input data from dataloader.

        Expects input dict from your aligned_dataset:
          {"A": fundus_tensor, "B": oct_tensor, "A_paths": ..., "B_paths": ...}
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device) # fundus tensorleri model içinde saklar (batch şeklinde)
        self.real_B = input["B" if AtoB else "A"].to(self.device) # oct tensorleri model içinde saklar (batch şeklinde)
        self.image_paths = input["A_paths" if AtoB else "B_paths"] # Modelin input görüntüsünün (fundus görüntü) dosya yolunu saklıyor.

    def forward(self):
        """
        Forward pass: fake_B = G(real_A)
        """
        self.fake_B = self.netG(self.real_A) # oluşturulan generatora fundus tensor batchi verilip karşılığında fake oct tensor batchi oluşturulur

    def backward_D(self): 
        """
        Discriminator loss:
          L_D = 0.5 * (GAN(D([A, B]), True) + GAN(D([A, G(A)]), False))
        """
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), dim=1) # concat fonksiyonu ile real_A ve fake_B birleştirilir
        pred_fake = self.netD(fake_AB.detach()) # detachlanır çünkü Dnin ağırlıkları güncellenirken G gradient almasın 
        # bir skor tahmin haritası oluşturur
        # mesela 30x30 luk patch oluştu bu her patch için bir doğruluk matrisi oluşturur ör; [0.8, 0.9, 0.7,...]
        self.loss_D_fake = self.criterionGAN(pred_fake, False) # target_is_real= False
        # target_tensor=0 alınır, buna göre patch patch loss hesaplanır

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True) # target_is_real = True 
        # target_tensor=1 alınır, buna göre patch patch loss hesaplanır

        # Combined
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 # 2 loss birleşir
        self.loss_D.backward() # disc için gradient hesaplanır

    def backward_G(self): # generatorı eğitmek için loss hesaplayıp gradient üretir
        # burada pred_real yoktur çünkü generatorın amacı fake_B yi real_B ye benzettirip disci kandırmaktır
        """
        Generator loss (your project formulation):
          L_G = lambda_adv * L_adv + lambda_L1 * L1

        where:
          L_adv = GAN(D([A, G(A)]), True)
          L1    = |G(A) - B|_1
        """
        fake_AB = torch.cat((self.real_A, self.fake_B), dim=1) 
        pred_fake = self.netD(fake_AB) # bu sefer detach yok! gradientin D üzerinden fake_B ye oradan da netG ye akmasını istiyoruz

        # Adversarial term # amaç; disci kandırmak --> D(fake_AB) --> real desin 
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_adv # fake görüntü gerçekmiş gibi görünsün

        # L1 term # amaç; üretilen octnin gerçek octye ne kadar benzediğini ölçmek
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 

        # Total
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self): # batch geldiği zaman sırayla D ve G yi güncelliyor
        """
        One optimization step: update D then G.
        """
        # Forward
        self.forward() # pseduo_oct batch üretimi 

        # Update D 
        self.set_requires_grad(self.netD, True) # BaseModel içinde tanımlanmıştır / gradient hesabı
        self.optimizer_D.zero_grad() # bi önceki adımdan kalan (önceki batchden kalan) D gradientleri temizle 
        self.backward_D() # D loss hesaplanır 
        self.optimizer_D.step() # D nin ağırlıkları güncellenir hesaplanan gradientlere göre 

        # Update G
        self.set_requires_grad(self.netD, False) # disc dondurulur böylece gradient sadece generatora gider
        self.optimizer_G.zero_grad() 
        self.backward_G() # G loss hesaplanır --> GAN loss = true olsun(kandırılır), L1 loss ile fake_B = real_B yapılmaya çalışılır
        self.optimizer_G.step()


        # requires_grad= true --> gradient hesaplanır
        # requires_grad= false --> gradient hesaplanmaz
