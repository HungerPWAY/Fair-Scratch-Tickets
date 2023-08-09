import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from time import sleep
from PIL import Image
from torchvision import transforms

if len(sys.argv) != 4:
    print("Usage: python simulate.py lambda_fair formulation GPU_index")
    exit(1)
lam_fair = float(sys.argv[1])
form = sys.argv[2]
gpui = int(sys.argv[3])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CelebA(root='./data', split='train', target_type='attr', transform=preprocess)
validset = torchvision.datasets.CelebA(root='./data', split='valid', target_type='attr', transform=preprocess)
testset  = torchvision.datasets.CelebA(root='./data', split='test',  target_type='attr', transform=preprocess)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
#validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True, num_workers=2)
#testloader  = torch.utils.data.DataLoader(testset,  batch_size=32, shuffle=False, num_workers=2)

#attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False) #pretrained = True if you want
model.fc = nn.Linear(512, 1, bias=True)
#model.load_state_dict(torch.load("baseline"))

#ti = attrs.index("Smiling")
#si = attrs.index("Male")

#(Pmale, Pfem) = (trainset.attr[:,si].float().mean(), 1 - trainset.attr[:,si].float().mean())

ploss = nn.BCEWithLogitsLoss()
'''
if form == "logistic":
    def floss(outputs, sens_attr):
        return -lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[sens_attr]).sum()/Pmale + F.logsigmoid(-outputs[~sens_attr]).sum()/Pfem)
elif form == "hinge":
    baseline = torch.tensor(0.).to(device)
    def floss(outputs, sens_attr):
        return lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[sens_attr]).sum()/Pmale + torch.max(baseline,1+outputs[~sens_attr]).sum()/Pfem)
else:
    def floss(outputs, sens_attr):
        return lam_fair/outputs.shape[0] * (-outputs[sens_attr].sum()/Pmale + outputs[~sens_attr].sum()/Pfem)
'''
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

device = torch.device("cuda:" + str(gpui) if torch.cuda.is_available() else "cpu")
model.to(device)
torch.cuda.empty_cache()

def calc_loss(data):
    #inputs, labels = data
    #nputs, labels, sens_attr = inputs.to(device), labels[:,ti].float().to(device), labels[:,si].bool().to(device)
    optimizer.zero_grad()
    outputs = model(inputs).reshape(-1)
    loss = ploss(outputs, labels) #+ floss(outputs, sens_attr)
    loss.backward()
    preds = (outputs >= 0).float()
    #unfairness = torch.tensor([preds[ sens_attr].sum(), preds[ sens_attr].shape[0],
    #                           preds[~sens_attr].sum(), preds[~sens_attr].shape[0]]) #msmiling, m, fsmiling, f
    return ((labels == preds).float().mean(), loss, unfairness)


print_every = 200
valid_batches = 32
for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    running_acc = 0.0
    running_unfair = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        (acc, loss, unfair) = calc_loss(data)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_acc += acc.item()
        running_unfair += unfair

        if i % print_every == (print_every - 1):    # print every 200 mini-batches
            print('[%d, %5d]' % (epoch + 1, i + 1))
            valid_loss = 0.0
            valid_acc = 0.0
            valid_iter = iter(validloader)
            for vi in range(valid_batches):
                (new_valid_acc, new_valid_loss, _) = calc_loss(next(valid_iter))
                valid_loss += new_valid_loss
                valid_acc += new_valid_acc.item()
            scheduler.step(valid_loss)
            print('Training Accuracy: %.3f, Validation Accuracy: %.3f, Unfairness: %.3f' % (running_acc / print_every,
                                                        valid_acc/valid_batches,
                                                        running_unfair[2]/running_unfair[3] -
                                                        running_unfair[0]/running_unfair[1]))
            print('Training Loss: %.3f, Validation Loss: %.3f' % (running_loss / print_every, valid_loss/valid_batches))
            print('')
            running_loss = 0.0
            running_acc = 0.0
            running_unfair = 0.0
            sleep(1)

print('Finished Training')

torch.save(model.state_dict(), form + str(lam_fair))