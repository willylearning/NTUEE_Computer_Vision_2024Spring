import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
        # each image size is 32 x 32
        self.cnn = nn.Sequential(
            # kernel_size=3 <=> kernel_size=(3,3)
            # after nn.Conv2d => output size = (input size + 2*padding - kernel size + 1)
            # 1st layer
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64 x 16 x 16 
            # 2nd layer
            nn.Conv2d(64, 128, 3, 1, 1), # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 x 8 x 8   
            # 3rd layer
            nn.Conv2d(128, 256, 3, 1, 1), # 256 x 8 x 8 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256 x 4 x 4 
        )
        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1024), # nn.Linear(in_features, out_features)
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        mynet_total_params = sum(p.numel() for p in self.cnn.parameters())
        # print(mynet_total_params)

    def forward(self, x):
        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)
        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        
        # 1. reduce the kernel size, stride of the first convolution layer
        # (original ResNet18's 1st layer : kernel_size=7, stride=2, padding=3)
        self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) 
        # 2. remove the first maxpool layer (i.e. replace with Identity())
        self.resnet.maxpool = Identity()
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10) # self.resnet.fc.in_features=512

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        resnet18_total_params = sum(p.numel() for p in self.resnet.parameters())
        # print(resnet18_total_params)

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)