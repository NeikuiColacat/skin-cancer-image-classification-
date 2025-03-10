import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
from torch import nn

def get_model(model_name: str):
    model = None
    use_pretrained = True
    num_classes = 7

    if model_name == 'resnet_pret':
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet_pret':
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    ###addtionally
    elif model_name == 'mobilenet_v3':
        model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'effiecentnet':
        model_ft = models.efficientnet_b1(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    ##
    else:
        print("Invalid model name, choose between 'resnet_pret', 'densenet_pret'.")
        exit()
    return model_ft

model_name = 'mobilenet_v3'
model = get_model(model_name)

model.load_state_dict(torch.load('mobilenetv3_pret.pth'))
model.eval()

norm_means = [0.77148203, 0.55764165, 0.58345652]
norm_std = [0.12655577, 0.14245141, 0.15189891]

img_h, img_w = 224, 224
val_test_transform = transforms.Compose([transforms.Resize((img_h,img_w)), transforms.ToTensor(),
                                        transforms.Normalize(norm_means, norm_std)])

image_path = 'infer_test/ISIC_0024525.jpg'
image = Image.open(image_path)
image = val_test_transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    scores = model(image)
    _ , pre = torch.max(scores,dim=1)
    print(pre)
    print(scores)
