import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'



arch_vgg16= [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' ]

class VGG16(nn.Module):
    def __init__(self , in_chn , num_class):
        super().__init__()
        self.in_channels = in_chn 
        self.num_class = num_class  
        self.conv_block = self.create_conv_block(arch_vgg16)   
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7*7*512 , out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096 , out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096 , out_features=self.num_class)
        )

    def forward(self,x):
        x = self.conv_block(x)
        return self.fc(x)
    def create_conv_block(self, architecture):
        in_chnnel = self.in_channels
        
        layers = []

        for x in architecture:
            if type(x) == int:
                out_channels = x 
                layers += [
                    nn.Conv2d(in_channels=in_chnnel, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                    nn.ReLU()       
                ]
                in_chnnel = x
            
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))]

        return nn.Sequential(*layers)

img = torch.randn(size=(1,3,224,224)).to(device)
model = VGG16(in_chn=3, num_class=10).to(device)
y = model.forward(img)
print(y.shape)
