import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class Yolopretrained_resnet(nn.Module):
    def __init__(self, backbone, g=7, b=2, c=1, **kwargs):
        super(Yolopretrained_resnet, self).__init__()
        self.g = g
        self.b = b
        self.c = c
        self.backbone = backbone
        self.head = nn.Sequential(

            nn.Linear(1000, 1000),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, self.g * self.g * (self.c + self.b * 5)),

        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class Yolopretrained_vgg16(nn.Module):
    def __init__(self, backbone, g=7, b=2, c=1, **kwargs):
        super(Yolopretrained_vgg16, self).__init__()
        self.g = g
        self.b = b
        self.c = c
        self.backbone = backbone
        self.detection = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.g * self.g * (self.c + self.b * 5)),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.detection(features)


#             return self.detection(features)




class CNN(nn.Module):
    def __init__(self,in_channels, out_channels, **kwargs):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.bachnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lrelu(self.bachnorm(self.conv(x)))




class Yolov1(nn.Module):
    def __init__(self, architecture, in_channels=3, **kwargs):
        super(Yolov1,self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fc = self._create_fc(**kwargs)



    def forward(self,x):
        x = self.darknet(x)
        # return torch.flatten(x,start_dim=1)
        return self.fc(torch.flatten(x, start_dim=1))



    def _create_conv_layers(self,architecture):
        in_channels = self.in_channels
        layers = []

        for layer in architecture:
            if isinstance(layer, tuple):
                layers.append(CNN(in_channels=in_channels,
                                  out_channels=layer[1],
                                  kernel_size=layer[0],
                                  stride=layer[2],
                                  padding=layer[3],
                                  )
                              )

                in_channels = layer[1]
            elif isinstance(layer, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif isinstance(layer, list):
                conv1 = layer[0]
                conv2 = layer[1]
                repeat = layer[2]

                for _ in range(repeat):
                    layers.append(CNN(in_channels=in_channels,
                                      out_channels=conv1[1],
                                      kernel_size=conv1[0],
                                      stride=conv1[2],
                                      padding=conv1[3],
                                      )
                                  )
                    layers.append(CNN(in_channels=conv1[1],
                                      out_channels=conv2[1],
                                      kernel_size=conv2[0],
                                      stride=conv2[2],
                                      padding=conv2[3],
                                      )
                                  )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)


    def _create_fc(self, grid_size, num_boxes, num_classes):
        g, b, c = grid_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 496),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(496, g*g*(c+b*5))  #(g*g*(c+b*5) = 7*7*11
        )
