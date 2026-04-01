# networks.py (minimal pix2pix for fundus -> pseudo-OCT)
import os
import functools
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


###############################################################################
# Helper layers / functions
###############################################################################

class Identity(nn.Module): # girdi neyse çıktısını da aynısı olarak verir.. norm_type=none olduğu zaman devreye girer
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"): # modelde kullanılacak normalization seçilir.default= instance
    """
    norm_type: batch | instance | none
    """
    if norm_type == "batch": # .partial --> num_features değerini şimdilik bilmediğimiz için şimdilik tutuyor sonra verirsin diyor
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True) # BatchNorm2d --> batch içindeki (mini-batch) istatistiklerle normalize eder
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False) # InstanceNorm2d --> her görüntüyü ayrı ayrı normalize eder
    elif norm_type == "none": 
        def norm_layer(_num_features):
            return Identity() # normalizasyon yerine Identity() döndürür
        return norm_layer
    else:
        raise NotImplementedError(f"Normalization layer [{norm_type}] is not found")


def get_scheduler(optimizer, opt): # optimizer ağırlıkları güncellerken kullandığı öğrenme oranı (lr)
    # scheduler ise zamanla bu lr yi değiştirir
    """
    opt should contain:
      - lr_policy: linear | step | plateau | cosine
      - ** pix2pixde en çok kullanılan -> (linear) epoch_count, n_epochs, n_epochs_decay --> ilk opt.n_epochs boyunda lrsabit sonrasında opt.n_epochs_decay boyunca yavaş yavaş 0 a lineer düşer
      - (step) lr_decay_iters --> Her lr_decay_iters epoch’ta bir LR’yi 0.1 ile çarp
    """
    if opt.lr_policy == "linear": # bu fonksiyon opt.lr_policy e göre scheduler seçer
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == "step":
        return lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    elif opt.lr_policy == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )

    elif opt.lr_policy == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)

    else:
        raise NotImplementedError(f"learning rate policy [{opt.lr_policy}] is not implemented")


def init_weights(net, init_type="normal", init_gain=0.02): # bu fonksiyon modelin içindeki katmanların başlangıç ağırlıklarını (init) ayarlar
    """
    init_type: normal | xavier | kaiming | orthogonal --> init yöntemleri 
    """
    def init_func(m): # modeldeki her modül/katman için çağırılır 
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1): # eğer katman conv veya linear ise ağırlıklarını initialize et
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")

            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func) # model içindeki tüm alt katmanları otomatik dolaşır


def init_net(net, init_type="normal", init_gain=0.02): # model cihaza taşınır ardından init_weights ı çağırır initialization gerçekleşir
    """
    Moves model to GPU if available, then initializes weights.
    Supports single GPU and (optional) DDP local rank.
    """
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    else:
        print("Initialized on CPU")

    init_weights(net, init_type, init_gain)
    return net


###############################################################################
# Factory: define Generator / Discriminator
###############################################################################

def define_G( # modeli oluşturan fonksiyon 
    input_nc,
    output_nc,
    ngf=64,
    netG="unet_256",
    norm="instance",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
):
    """
    Minimal choice: only unet_256
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG != "unet_256":
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized. Use 'unet_256'.")

    net = UnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        num_downs=8,   # for 256x256
        ngf=ngf,
        norm_layer=norm_layer,
        use_dropout=use_dropout,
    )
    return init_net(net, init_type, init_gain)


def define_D(
    input_nc,
    ndf=64,
    netD="basic",
    n_layers_D=3, # patch büyüklüğü, ne kadar büyükse o kadar daha büyük bağlam görür 
    norm="instance",
    init_type="normal",
    init_gain=0.02,
):
    """
    netD: basic | n_layers
    - basic = PatchGAN with n_layers=3
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":
        net = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":
        net = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f"Discriminator model name [{netD}] is not recognized. Use 'basic' or 'n_layers'.")

    return init_net(net, init_type, init_gain) # modeli GPU ya yollar, ağırlıkları initialize eder  


###############################################################################
# Loss
###############################################################################

class GANLoss(nn.Module):
    """
    GAN objectives: vanilla | lsgan | wgangp
    Note: If using vanilla, Discriminator last layer should NOT have sigmoid.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss() # ör; D output = 0.8, target=1, loss = (0.8 - 1)^2
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss() # Binary Cross Entropy
        elif gan_mode == "wgangp": # WGAN farklı çalıştığı için farklı hesaplanıyor 
            self.loss = None
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")

    def get_target_tensor(self, prediction, target_is_real: bool):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction) # Target tensoru, discriminator çıktısının boyutuna (shape’ine) genişletiyoruz.

    def forward(self, prediction, target_is_real: bool):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real) # Target tensor oluşturulur ör; prediction = [[0.6,0.7,...]], target = [[1,1,...]]
            return self.loss(prediction, target_tensor) # Loss hesaplanır

        # wgangp
        return -prediction.mean() if target_is_real else prediction.mean()


###############################################################################
# Generator: U-Net
###############################################################################

class UnetGenerator(nn.Module): # blokları birleştirir
    """
    U-Net generator from pix2pix (recursive construction).
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False): # num_downs= down sampling kaç kez olacak -> u-net derinliğini belirler 
        super().__init__()

        # innermost
        unet_block = UnetSkipConnectionBlock(
            outer_nc=ngf * 8, inner_nc=ngf * 8, input_nc=None, # en derin katmanlarda kanal sayısı 512 
            submodule=None, innermost=True, norm_layer=norm_layer
        )

        # middle blocks with ngf*8 --> en dipteki bloka bir middle blok sarılıyor
        for _ in range(num_downs - 5): # submodule=unet_block; en iç blok alınır onun dışına başka bir blok sarılır 
            unet_block = UnetSkipConnectionBlock(
                outer_nc=ngf * 8, inner_nc=ngf * 8, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )

        # gradually reduce filters --> dış seviyeler ekleniyor # middle block
        # blok içinde conv’ler yaklaşık bu kanal seviyelerinde feature map üretiyor.                    
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer) # o bloktaki conv katmanlarının kanal sayıları 256 ve 512 civarında çalışıyor
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf,     ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        # outermost --> en dış blok ekleniyor
        self.model = UnetSkipConnectionBlock(
            outer_nc=output_nc, inner_nc=ngf, input_nc=input_nc,
            submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    """
    X ------------------- skip ----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """
    # ör 256 (outer_nc) → 512 (inner_nc) → submodule → 256 

    def __init__(  
        self,
        outer_nc, # bloğun çıkış kanal sayısı
        inner_nc, # bloğun iç tarafındaki kanal sayısı
        input_nc=None,
        submodule=None, # bloğun içinde bulunan daha derin u-net bloğu 
        outermost=False, # giriş görüntüsünü alan blok 
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost

        # bias rule: if InstanceNorm then conv bias True, else False
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias) # boyutu yarıya indirme işlemi 256x256 -> 128x128
        downrelu = nn.LeakyReLU(0.2, True) # aktivasyon ; encoder tarafında LeakyReLU kullanılır
        downnorm = norm_layer(inner_nc) # normalizasyon 

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost: # input image alır, final output üretir --> en dış blok # aslında hem giriş hem çıkış kapısı 
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1) # görüntüyü 256x256 ya geri getirme işlemi
            down = [downconv] 
            up = [uprelu, upconv, nn.Tanh()] # .Tanh() --> pix2pix çıktıyı [-1 1] aralığında üretir --> final image output üretimi
            model = down + [submodule] + up # [submodule] --> bu noktada daha derindeki U-net bloğunu çalıştır sonra geri dönünce "up" yap 

        elif innermost: # en dipteki blok 
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm] # upnorm ara feature mapler için
            model = down + up # daha derinde başka blok olmadığı için submodule yok 

        else: # bunların arasında kalan middle bloklar için;
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up 

        self.model = nn.Sequential(*model)

    def forward(self, x): # x ; o seviyedeki encoder feature (yani o bloğa gelen orijinal feature), self.model(x); down->submodule->up sonrası gelen decoder feature 
        if self.outermost: 
            return self.model(x) # outermost direkt döndürülür --> burada feature map birleştirip kanal sayısını büyütmek istenmiyor son görüntü üretilmek isteniyor
        # concat skip connection
        return torch.cat([x, self.model(x)], dim=1) # o seviyedeki encoder feature ve decoder feature birleştirilir


'''
decoder+encoder concat şematik gösterim;
Encoder
   ↓
feature x
   ↓
   ├──────────── skip ───────────┐
   ↓                             │
 deeper block                    │
   ↓                             │
 Decoder output                  │
   ↓                             │
 concat(x , decoder_output) ◄───┘


middle blokta;
downrelu -> downconv -> downnorm
    -> submodule ->
uprelu -> upconv -> upnorm
'''

###############################################################################
# Discriminator: PatchGAN
###############################################################################

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator (outputs a prediction map).
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d) 
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        kw = 4 # model görüntüye 4x4 parçalar halinde bakıyor -> her baktığı yerden --> tek sayı feature çıkarma işlemi 
        padw = 1 # kernel görüntünün kenarına geldiğinde taşma olayları olmaması adına görüntünün etrafına 0 pixel eklenir 
        # stride: kernel her adımda kaç pixel kayıyor --> 2 pixel atlayarak ilerler downsampling yapar --> çok büyük olursa output resolution küçülür
        # bu üçü kullanılarak output size oluşturulur 

        sequence = [ # disc katmanları burada sırayla tutulacak
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), # ilk layer --> normalizasyon yok raw image feature çıkar [batch,4,256,256] > [batch,64,128,128]
            nn.LeakyReLU(0.2, True), # activation fonk
        ]

        nf_mult = 1 # filter sayısının nasıl büyüyeceğini kontrol eden değişkenler
        nf_mult_prev = 1

        # layers with stride=2 # discin orta katmanları oluşturuluyor, her turda kanal sayısı arttırılıp görüntü boyutu yarıya indiriliyor
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), # n=1 ; Conv2d(64*1, 64*2), n=2 Conv2d(64*2, 64*4)
                norm_layer(ndf * nf_mult), 
                nn.LeakyReLU(0.2, True),
            ]

        # last conv with stride=1
        nf_mult_prev = nf_mult # 4, giriş kanalı 64*4= 256
        nf_mult = min(2**n_layers, 8) # -> min(8,8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), # nn.Conv2d(64*4, 64*8) >>çıkışta 512 kanal üretilir
            norm_layer(ndf * nf_mult), # norm_layer(512) >> conv sonrası çıkan 512 feature map normalize edilir
            nn.LeakyReLU(0.2, True), 
        ]

        # output 1-channel prediction map --> tek kanallı bir çıktı haritası üretilir (512 -> 1) >> çıktı: [batch,1,30,30]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # sequence += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)] --> output conv

        self.model = nn.Sequential(*sequence) # katmanların hepsi sırayla çalışıyor

    def forward(self, x):
        return self.model(x) # x (input tensor) → self.model (bütün disc katmanlarından geçer) → patch map 