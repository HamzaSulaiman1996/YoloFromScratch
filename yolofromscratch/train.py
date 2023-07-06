from model import Yolopretrained_resnet
from dataset import YOLODataset
from loss import YoloLoss
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = os.path.join(os.getcwd(), 'Project')
transform = A.Compose([
            A.Resize(448,448),
            A.ShiftScaleRotate(rotate_limit=30,p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.2),
            ],
            bbox_params=A.BboxParams(format='yolo',
                                     min_visibility=0.4,
                                     label_fields=[],
                                    ),
)


dataset = YOLODataset(root=ROOT,
                      transform=transform,
                     )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                         shuffle=True,
                                         )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
for param in backbone.parameters():
    param.requires_grad = False

model = Yolopretrained_resnet(backbone)
model = model.to(device)

if os.path.exists(f'{ROOT}/full_model_resnet.pt'):
    model = torch.load(f'{ROOT}/full_model_resnet.pt')

loss_fn = YoloLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

model = model.train()
for epoch in range(50):

    mean_loss = []
    mean_classloss = []
    mean_boxloss = []
    mean_objloss = []
    mean_noobj_loss = []
    for img, labels in tqdm(dataloader):
        images = img.permute(0, 3, 1, 2).to(device)
        out = model(images)
        #         loss = loss_fn(out.to('cpu'),labels)
        class_loss, object_loss, box_loss, no_object_loss, loss = loss_fn(out, labels.to(device))
        # print(f"Class Loss:{class_loss} Obj Loss:{object_loss} Box Loss:{box_loss} Loss:{loss}",
        #       # end="\r",
        #       )

        mean_loss.append(loss.detach().item())
        mean_classloss.append(class_loss.item())
        mean_boxloss.append(box_loss.item())
        mean_objloss.append(object_loss.item())
        mean_noobj_loss.append(no_object_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1} \
    Class Loss:{sum(mean_classloss) / len(mean_classloss)} Object Loss:{sum(mean_objloss) / len(mean_objloss)} \
    BoxsLoss:{sum(mean_boxloss) / len(mean_boxloss)} NoObjLoss:{sum(mean_noobj_loss) / len(mean_noobj_loss)}")

    print(f"\nMean loss was {sum(mean_loss) / len(mean_loss)}")

    if epoch % 5 == 0:
        torch.save(model, f'{ROOT}/full_model_resnet_drone.pt')