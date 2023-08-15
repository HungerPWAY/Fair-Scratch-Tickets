import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.vision import VisionDataset
import numpy as np


class CelebADataset(VisionDataset):
    

    def __init__(self, root, split='train', transform=None, target_transform=None):
        
        super(CelebADataset, self).__init__(root, transform=transform,
                                             target_transform=target_transform)
        
        self.split = split
        

        self.dataset = torchvision.datasets.CelebA(
        root = root,
        split = split,
        target_type ='attr',
        transform = transform
        )


        attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()
        self.ti = attrs.index("Smiling")
        self.si = attrs.index("Male")
        #print(self.dataset.attr[:self.si].shape)
        (Num_male_smile, Num_fem_smile) = (torch.count_nonzero((self.dataset.attr[:,self.si].bool() & self.dataset.attr[:,self.ti].bool()).int()),
                torch.count_nonzero((~self.dataset.attr[:,self.si].bool() & self.dataset.attr[:,self.ti].bool()).int()))
        #Pmatrix  = np.array(data.train_dataset.attr[:,si].float().mean(), 1 - data.train_dataset.attr[:,si].float().mean())
        (Num_male, Num_fem) = (torch.count_nonzero(self.dataset.attr[:,self.si].int()), len(self.dataset) - torch.count_nonzero(self.dataset.attr[:,self.si].int()))
        #self.Nmatrix = np.array([[Num_male_smile, Num_male - Num_male_smile],
        #                    [Num_fem_smile, Num_fem - Num_fem_smile]])
        self.Nmatrix = np.array([[Num_fem - Num_fem_smile, Num_fem_smile],
                                   [Num_male - Num_male_smile, Num_male_smile]])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        return self.dataset[index][0], self.dataset.attr[index,self.ti], self.dataset.attr[index,self.si]


class CelebA:
    def __init__(self, args):
        super(CelebA, self).__init__()

        #data_root = os.path.join(args.data, "celeba")
        data_root = args.data
        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = CelebADataset(
            root=data_root,
            split='train',
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        print(len(train_dataset))
        self.train_Nmatrix = train_dataset.Nmatrix
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        val_dataset = CelebADataset(
            root=data_root,
            split='valid',
            transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(), normalize]),
        )
        self.val_Nmatrix = val_dataset.Nmatrix
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

        test_dataset = CelebADataset(
            root=data_root,
            split='test',
            transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(), normalize]),
        )
        self.test_Nmatrix = test_dataset.Nmatrix
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )