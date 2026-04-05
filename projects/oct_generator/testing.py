from data.aligned_dataset import AlignedDataset
from options.base_options import BaseOptions
import os

opt = BaseOptions().parse()

opt.phase = "test"
opt.isTrain = False

print("Dataroot:", opt.dataroot)
print("Expected A:", os.path.join(opt.dataroot, opt.phase, opt.dirA))
print("Expected B:", os.path.join(opt.dataroot, opt.phase, opt.dirB))

dataset = AlignedDataset(opt)

print("DATASET SIZE:", len(dataset))

for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(sample["A_paths"], sample["B_paths"])