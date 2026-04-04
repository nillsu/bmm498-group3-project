import time # zaman ölçmek için kullanılır --> 1 epoch ne kadar sürdü?
import torch # Pytorch kütüphanesi --> tensor işlemleri, GPU/CPU seçimi, modeli çalıştırma
from options.base_options import BaseOptions # dataroot, batch_size, n_epochs, lr gibi parametreleri okur
from data.aligned_dataset import AlignedDataset # fundus-oct klasörünü okuma, isimleri eşleştirme, her seferinde fundus-oct çifti verme
from torch.utils.data import DataLoader # dataseti batch batch verir -> DataLoader hazır Pytorchun bir sınıfıdır
from models import create_model # pix2pix modelini çağırır, generator/disc kurar
import copy


def validate(model, dataloader): # eğitim bittikten sonra değil her epoch sonunda val dataset üzerinden loss hesaplamak
    model.eval() # model evaluation moduna alınır
    total_loss = 0.0 # val datasından gelen tüm lossları toplayacağımız değişken

    with torch.no_grad(): # validation sırasında gradient/ağırlık güncellenmez sadece performans ölçümü 
        for data in dataloader: # burada val dataseti batch batch gelir
            model.set_input(data) # batchi modele verir
            model.forward() # generator çalışır --> fundus girer-> fake OCT üretilir sadece ileri geçiş

            # G validation losses must be computed explicitly here
            fake_AB = torch.cat((model.real_A, model.fake_B), dim=1)
            pred_fake = model.netD(fake_AB)

            loss_G_GAN = model.criterionGAN(pred_fake, True) * model.opt.lambda_adv
            loss_G_L1 = model.criterionL1(model.fake_B, model.real_B) * model.opt.lambda_L1

            loss = loss_G_GAN + loss_G_L1
            total_loss += loss.item() # bütün batchlerin lossu toplanır


    # switch back to train mode
    model.netG.train() # validation bitti, generator tekrar training moduna alınır -->eğitim devam eder
    if hasattr(model, "netD"): # modelde disc varsa (trainde olur)
        model.netD.train() # disc tekrar öğrenmeye hazır olsun 

    return total_loss / len(dataloader) # ortalama loss alınır

'''
model.loss_G_L1 -> Generator’ın gerçek OCT ile üretilen OCT arasındaki L1 kaybı / piksel bazında ölçüm
model.loss_G_GAN -> Generator’ın discriminator’ı kandırma başarısıyla ilgili adversarial loss / görüntü ne kadar gerçekçi?
loss = model.loss_G_L1 + model.loss_G_GAN -> Toplam generator validation loss’u alınır
total_loss += loss.item() -> Her batch için çıkan sayısal loss değerini toplar
.item() tensorü normal Python sayısına çevirir

'''

if __name__ == "__main__": 

    # OPTIONS
    opt = BaseOptions() # ayar objesi kurulur. henüz boş gibi düşün
    opt.isTrain = True # bu kod training modunda çalışıyor 
    opt = opt.parse() # terminal veya default değerlerden ayarları okur-> ve bunları opt içine koyar

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU var mı diye bakar, eğer varsa-> cuda 
    opt.device = device # sonrasında bu bilgi opt içine eklenir
    print(f"Using device: {device}") 

    # =========================
    # TRAIN DATASET
    # =========================
    opt.phase = "train" # dataset "train" klasörünü kullansın--> dataroot/train/...
    train_dataset = AlignedDataset(opt) # Train dataset nesnesi oluşturulur--> fundus-oct çiftleri bulunur, eşleştirilir, kullanıma hazır hale getirilir

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size, # bir seferde kaç görüntü döndürülecek (4)
        shuffle=True, # train datasını karıştırır -> model hep aynı sırada görüp ezberlemesin diye 
        num_workers=opt.num_threads, # veri yüklemek için kaç alt işlem var ->hız için
    )

    print(f"Number of training images = {len(train_dataset)}") # train datasetinde kaç çift var yazdırılır

    # =========================
    # VALIDATION DATASET
    # =========================
    opt.phase = "val" # val klasörünü kullan --> dataroot/val/...
    val_dataset = AlignedDataset(opt)

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False, # validationda öğrenme yoktur sadece ölçüm yapılır o yüzden false denilir
        num_workers=opt.num_threads,
    )

    print(f"Number of validation images = {len(val_dataset)}") # val dataset kaç görüntü içeriyor yazdırır

    # MODEL
    model = create_model(opt) # model oluşturulur
    model.setup(opt) # optimizer, scheduler kurar, checkpoint yükler, ağları hazırlar 

    # =========================
    # EARLY STOPPING PARAMS --> early stopping için başlangıç ayarları yapılıyor
    # =========================
    best_val_loss = float("inf") # Başlangıçta en iyi val loss sonsuz olsun diyorsun. Yani ilk gelen gerçek val loss bundan küçük olacağı için “en iyi” kabul edilir.
    patience = 10 # validation loss kötüleşince kaç epoch daha sabredecek?
    counter = 0 # şu ana kadar kaç kez üst üste kötüleştiğini sayacak 

    # TRAIN LOOP
    total_iters = 0 # kaç görüntü işlendiğini takip eder

    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1): # ana eğitim döngüsü

        epoch_start_time = time.time() # epochun başlangıç anını kaydeder

        # ================= TRAIN =================
        for i, data in enumerate(train_loader): # train datadaki her batch için çalışır

            total_iters += opt.batch_size # her batchde işlenen görüntü sayısını ekliyor

            model.set_input(data) # o batchi modelin içine koyar
            model.optimize_parameters() # asıl trainingin olduğu yer --> forward->disc loss hesaplama->disc update->gen loss hesapla->gen update

            if total_iters % 100 == 0: # her 100 iterasyonda bir mevcut lossları yazdırır
                losses = model.get_current_losses()
                print(f"[Epoch {epoch} | Iter {total_iters}] {losses}")

        # ================= VALIDATION =================
        val_loss = validate(model, val_loader) # epoch içindeki train bittikten sonra model val dataset üzerinde test edilir-> ortalama val loss hesaplanır 
        print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}") # yazdırılır

        # ================= EARLY STOPPING =================
        if val_loss < best_val_loss:  # Eğer bu epoch’taki val loss, şimdiye kadarki en iyi val loss’tan küçükse:
            best_val_loss = val_loss # güncellenir
            counter = 0 # kötüleşme sayacı sıfırlanır

            print("Saving BEST model") # val loss iyileştiyse en iyi modeli kaydet
            model.save_networks("best")

        else: # eğer iyileşmediyse 
            counter += 1 # sayaç 1 artar
            print(f"EarlyStopping Counter: {counter}/{patience}") 

        if counter >= patience: # üst üste 5 epoch iyileşme olmadıysa 
            print("EARLY STOPPING TRIGGERED") # training durur -> overfittingi önlemek için!!
            break

        # LR update
        model.update_learning_rate() # her model sonunda scheduler çalışır 
        # --> bu genelde learning rate i azaltır ve eğitimi daha stabil hale getirir

        # regular save --> her 5 epochda bir modeli kaydeder --> yani epoch 5, epoch 10, epoch 15 gibi checkpoint oluşur 
        if epoch % 5 == 0: 
            print(f"Saving model at epoch {epoch}")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} \t Time Taken: {time.time() - epoch_start_time:.2f} sec") # epoch kaç saniye sürdü yazdırır