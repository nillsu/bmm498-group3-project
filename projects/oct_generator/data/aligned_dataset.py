import os
from data.base_dataset import BaseDataset, get_transform  # get_params yok 
from data.image_folder import make_dataset
from PIL import Image


def _key_from_filename(fname: str) -> str:
   
    base = os.path.splitext(os.path.basename(fname))[0]
    '''
    os.path.basename(fname) >> tam pathden sadece dosya adını alır 
    ör; input: C:\...\1221_OD_000000_fundus.jpg, output: 1221_OD_000000_fundus.jpg

    os.path.splitext(...) >> dosyanın uzantısını ayırır 
    ör; input: 1221_OD_000000_fundus.jpg output: (1221_OD_000000_fundus, .jpg)

    [0] >> sadece uzantısız kısmı alır ör; base= "1221_OD_000000_fundus"
    '''

    for suf in ["_fundus", "_oct", "_OCT", "_bscan", "_Bscan", "_oct_v2"]:  # Eğer dosya adı bunlardan biriyle bitiyorsa, onu kırp.
        if base.endswith(suf):
            base = base[: -len(suf)] # sondan suffix kadar karakteri keser, base= "1221_OD_000000"
            break
    return base


class AlignedDataset(BaseDataset): # AlignedDataset, BaseDataset’in kurallarına uymak zorunda
    """Paired dataset from two folders (A and B)."""

    # __init__ >> hazırlık yapar >> çiftleri bulup listeler 
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        """
        opt içinde; 
        dataroot
        phase (train/test/val)
        input_nc
        output_nc
        crop_size vs.
        Yani bütün ayarlar burada geliyor.
        """

        phase_dir = opt.phase # "train"
        if os.path.isdir(os.path.join(opt.dataroot, phase_dir + "_pseudo")): # os.path.isdir >> bu klasör gerçekten var mı sorgular 
            # opt.dataroot >> ana klasör : C:\ML\pseudo_oct model data set
            phase_dir = phase_dir + "_pseudo" # "train_pseudo"

        base_dir = os.path.join(opt.dataroot, phase_dir) # C:\ML\pseudo_oct model data set\train_pseudo

     
        a_name = getattr(opt, "dirA", "fundus_pse-model") 
        b_name = getattr(opt, "dirB", "oct_pse-model")

        self.dir_A = os.path.join(base_dir, a_name) # C:\ML\pseudo_oct model data set\train_pseudo/fundus_pse-model
        self.dir_B = os.path.join(base_dir, b_name) # C:\ML\pseudo_oct model data set\train_pseudo/oct_pse-model

        assert os.path.isdir(self.dir_A), f"A folder not found: {self.dir_A}" 
        assert os.path.isdir(self.dir_B), f"B folder not found: {self.dir_B}"

        A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size)) # self.dir_A klasörünün içindeki tüm resim dosyalarının yollarını topluyor
        '''
        train_pseudo/fundus_pse-model/
        1221_OD_000000_fundus.jpg
        1222_OD_000000_fundus.jpg

        o zaman A_paths şöyle bir liste oluyor; [ "C:/ML/.../1221_OD_000000_fundus.jpg", "C:/ML/.../1222_OD_000000_fundus.jpg"]
        '''
        B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size)) # oct klasöründeki tüm resimleri al

        # eşleştirme 
        A_map = {_key_from_filename(p): p for p in A_paths}  # for p in A_paths>> A_paths listesindeki her bir elemanı sırayla al
        B_map = {_key_from_filename(p): p for p in B_paths}
        '''
        A_map = {
        "1221_OD_000000": "C:/ML/.../1221_OD_000000_fundus.jpg",
        "1222_OD_000000": "C:/ML/.../1222_OD_000000_fundus.jpg"
        }
        '''
        keys = sorted(set(A_map.keys()) & set(B_map.keys()))  
        # .keys() >> anahtarı verir (yani mesela "1221_OD_000000")
        # & >> iki listede de bulunan ortak anahtarları al.
        # sorted >> sıraya koyar 

        if len(keys) == 0:
            raise RuntimeError(
                "No paired files found. Check naming (_fundus/_oct suffix) or folders."
            )

        self.pairs = [(A_map[k], B_map[k]) for k in keys] # fundus oct path çiftleri oluşturulur

        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        '''
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        '''

    # __getitem__ >> eğitim sırasında her seferinde bir çift verir 
    def __getitem__(self, index): # dataloaderın istediği index numaralı olan çifti verir
        A_path, B_path = self.pairs[index] # çiftin dosya yolunu seçiyor

        # A: fundus (RGB)
        A_img = Image.open(A_path).convert("RGB")

        # B: OCT (grayscale)
        B_img = Image.open(B_path).convert("L")  

        # same transform params for both
        # transform_params = None     # get_params(self.opt, A_img.size) # random crop bilgisi KAPALI 
        # davranışını opt üzerinden (içindeki parametrelere göre) belirler
        # başlangıç için flip/crop kapalı tutalım >> random parametre üretmicek >> transform_params geçersiz



        A_transform = get_transform(self.opt, grayscale=False) # transform_params bilgisi kaldırıldı 
        
        '''
        Yani get_transform şuna bakıyor:
        “Crop var mı?”
        “Flip var mı?”
        “Normalize nasıl?”
        “Resize kaç?”
        !!! Hepsini self.opt’tan öğreniyor !!!
        '''
        # ToTensor, normalize yapar (şimdilik flip/cropsuz)

        B_transform = get_transform(self.opt, grayscale=True)

        A = A_transform(A_img) # fundus tensor
        B = B_transform(B_img) # real oct tensor 

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return len(self.pairs)
