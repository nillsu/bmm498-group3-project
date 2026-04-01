import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name): # "pix2pix" gibi bir isim alıp buna karşılık gelen python dosyasını açıyor 
    model_filename = f"models.{model_name}_model" # dosya adını oluşturma 
    modellib = importlib.import_module(model_filename) # modülü import etme -> dosyayı aç içindekileri modellib'e yükle 

    target_model_name = model_name.replace("_", "") + "model"

    for name, cls in modellib.__dict__.items(): # import edilen dosyanın içindeki her şeyi gezer 
        if (
            name.lower() == target_model_name.lower()
            and isinstance(cls, type)
            and issubclass(cls, BaseModel) # BaseModel alt sınıf mı Kontrol
        ):
            return cls

    raise ValueError(f"Model {model_name} not found.")


def create_model(opt):
    model_class = find_model_using_name(opt.model) 
    model = model_class(opt)
    print(f"Model [{model_class.__name__}] created")
    return model

'''
get _option_setter fonksiyonu kaldırıldı

bu dosya;
--> "pix2pix" → Pix2PixModel mapping yapar
--> class’ı bulur
--> instance oluşturur

'''