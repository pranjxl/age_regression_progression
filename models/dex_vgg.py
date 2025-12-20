import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Base VGG model
# -------------------------
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.fc6 = nn.Linear(25088, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        # 101 age classes (0–100)
        self.fc8_101 = nn.Linear(4096, 101)

        if pool == 'max':
            pool_fn = nn.MaxPool2d
        else:
            pool_fn = nn.AvgPool2d

        self.pool1 = pool_fn(2, 2)
        self.pool2 = pool_fn(2, 2)
        self.pool3 = pool_fn(2, 2)
        self.pool4 = pool_fn(2, 2)
        self.pool5 = pool_fn(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        out = self.fc8_101(x)     # logits for 101 classes

        return out


# -------------------------
# DEX age regressor wrapper
# -------------------------
class DEX_VGG(nn.Module):
    """
    Wraps VGG age classifier and converts logits → predicted age.
    """
    def __init__(self):
        super(DEX_VGG, self).__init__()
        self.model = VGG()

        # Age labels 0..100
        self.register_buffer("ages", torch.arange(0, 101).float())

    def forward(self, x):
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)
        age = torch.sum(prob * self.ages.to(prob.device), dim=1)

        return age, logits
