import torch
import torch.nn as nn

__all__ = ["ResNet50", "ResNet101", "ResNet152"]


def Conv1(in_places, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_places,
            out_channels=places,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        ),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        layers = [
            nn.Conv2d(
                in_channels=in_places,
                out_channels=places,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=places,
                out_channels=places,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=places,
                out_channels=places * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(places * self.expansion),
        ]
        self.bottleneck = nn.Sequential(*layers)
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_places,
                    out_channels=places * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(places * self.expansion),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.bottleneck(x)
        if self.downsampling:
            res = self.downsample(x)
        out += res
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        assert len(blocks) >= 4
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_places=3, places=64)
        self.layers = [
            self.make_layer(in_places=64, places=64, block=blocks[0], stride=1),
            self.make_layer(in_places=256, places=128, block=blocks[1], stride=2),
            self.make_layer(in_places=512, places=256, block=blocks[2], stride=2),
            self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2),
        ]
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet50():
    return ResNet([3, 4, 6, 3])


def ResNet101():
    return ResNet([3, 4, 23, 3])


def ResNet152():
    return ResNet([3, 8, 36, 3])
