from torch.utils.data import Dataset
import json
from torchvision.datasets import ImageFolder
import sys
import torch
from ft_util.datasets import build_dataset, build_transform
from torchvision import transforms
import PIL.Image as Image
import numpy as np

class FilterDataset(Dataset):
    def __init__(self, dataset, spec_file, num_classes=712):
        self.root = dataset.root
        self.loader = dataset.loader
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

        dataset_spec = json.load(open(spec_file))
        self.classes = list(dataset_spec['class_names'].values())[:num_classes]

        self.samples = []
        #print(dataset.samples[:10])
        #print(len(dataset.samples))
        for path, label in dataset.samples:
            cls = path.split('/')[-2]
            if cls in self.classes:
                #self.sample.append((path, label))
                self.samples.append((path, self.classes.index(cls)))

        print('Use {} classes from original dataset as training set, {} images in total'.format(len(self.classes), len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #sample = np.asarray(sample)
        return sample, target

def url_norm(inp):
    func = transforms.ToTensor()
    #return 2 * (func(inp) - 0.5)
    return 2 * (inp - 0.5)

if __name__ == '__main__':

    normalize = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #url_norm,
    ])

    # first global crop
    global_transfo1 = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        #normalize,
    ])
 
    train_transform = global_transfo1
    ori_dataset = ImageFolder('/jizhi_data/ILSVRC2012_data/train', transform=train_transform)
    new_dataset = FilterDataset(ori_dataset, './dataset_spec.json')
    data_loader = torch.utils.data.DataLoader(
        new_dataset,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    for i, (data, target) in enumerate(data_loader):
        import pdb;pdb.set_trace()
        print(target)
