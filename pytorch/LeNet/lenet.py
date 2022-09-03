import torch 
import torch.nn as nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed=seed)



class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5),stride=(1,1))
        self.pool = nn.AvgPool2d(kernel_size=(2,2) , stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5),stride=(1,1))
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5 , inplace=False)   
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        print(f'this first con sahpe {x.shape}')
        x = self.pool(x)
        print(f'this first  pool shape {x.shape}')
        x = self.relu(self.conv2(x))
        print(f'this second con sahpe {x.shape}')
        x = self.pool(x)
        print(f'this second  pool shape {x.shape}')
        x = self.relu(self.conv3(x))
        print(f'this third con sahpe {x.shape}')
        x = x.reshape(x.shape[0] , -1)
        #x = x.view(x.shape[0] , -1)
        print(f'this reshape x {x.shape}')
        x = self.relu(self.fc1(x))
        print(f'this first FC sahpe {x.shape}')
        # x= self.drop(x)
        # print(f'this second dropout sahpe {x.shape}')
        x = F.softmax(self.fc2(x), dim=1)
        # print(f'this second FC sahpe {x.shape}')
        return x

img = torch.randn(size=(1,1,32,32))
model = Lenet()
y_pred = model.forward(img)
print(y_pred.shape)
