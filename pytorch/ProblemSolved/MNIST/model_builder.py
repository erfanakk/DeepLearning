
import torch
import torch.nn as nn
import torch.functional as F







torch.manual_seed(42)
class CNN_simple(nn.Module):
    def __init__(self , in_channels, num_class):
        super().__init__() 
        #img --> (1,28,28)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels= 8, kernel_size=(3,3) , stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3) , stride=(1,1), padding=(1,1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features= 16 * 7 * 7 , out_features=num_class)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.reshape(x.shape[0] , -1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x 
                   
class Lenet(nn.Module):
    def __init__(self , in_channels , num_class):
        super().__init__()
        #img --> (1,32,32)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5),stride=(1,1))
        self.pool = nn.AvgPool2d(kernel_size=(2,2) , stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5),stride=(1,1))
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_class)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5 , inplace=False)   
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        #print(f'this first con sahpe {x.shape}')
        x = self.pool(x)
        #print(f'this first  pool shape {x.shape}')
        x = self.relu(self.conv2(x))
        #print(f'this second con sahpe {x.shape}')
        x = self.pool(x)
        #print(f'this second  pool shape {x.shape}')
        x = self.relu(self.conv3(x))
        #print(f'this third con sahpe {x.shape}')
        x = x.reshape(x.shape[0] , -1)
        #x = x.view(x.shape[0] , -1)
        #print(f'this reshape x {x.shape}')
        x = self.relu(self.fc1(x))
        #print(f'this first FC sahpe {x.shape}')
        x= self.drop(x)
        # print(f'this second dropout sahpe {x.shape}')
        x = self.fc2(x)
        # print(f'this second FC sahpe {x.shape}')
        return x


if '__main__' == __name__:
    img = torch.rand(size=(1,1,28,28))
    model = CNN_simple(in_chn=1, n_out=10)
    print(model(img).shape)
    img = torch.rand(size=(1,1,32,32))
    model2 = Lenet(in_channels=1 , num_class=10)
    print(model2(img).shape)