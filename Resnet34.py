import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import time
'''
데이터를 가공해주는 부분
이 부분은 어떻게 처리하는 지에 따라 성능에도 크게 달라질 수 있다.
하지만, 어찌 처리해야할 지 아직 잘 모르겠다.
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#https://github.com/chenyuntc/pytorch-best-practice/blob/master/models/ResNet34.py
class BasicBlock(nn.Module):
    '''
    Figure 5.에 해당하는 Block을 구성
    '''
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.basicblock = nn.Sequential(
            #첫 경우에는 downsampling을 위해 stride=2로 함.
            #이를 위해 stride로 입력값을 줌.
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut

    def forward(self, x):
        output = self.basicblock(x)
        if self.shortcut != None:
            residual = self.shortcut(x)
        else:   
            residual = x
        output = output + residual
        return F.relu(output)

class ResNet34(nn.Module):
    '''
    전체 resnet-34layer 구성
    '''
    def __init__(self,num_class=10):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.conv2_pre = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_main = self._make_layer(in_channels=64, out_channels=64, num_blocks=3, kernel_size=3)
        self.conv3 = self._make_layer(in_channels=64, out_channels=128, num_blocks=4, kernel_size=3)
        self.conv4 = self._make_layer(in_channels=128, out_channels=256, num_blocks=6, kernel_size=3)
        self.conv5 = self._make_layer(in_channels=256, out_channels=512, num_blocks=3, kernel_size=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # CIFAR-10의 Class -> 10개이기에 linear의 결과를 10개로 해준다.
        # 논문에서는 이 부분이 1000이었음.
        self.fc = nn.Linear(512, num_class)
        

    def _make_layer(self, in_channels, out_channels, num_blocks, kernel_size=3):

        shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        layer = []
        #첫 경우에는 downsampling을 위해 stride=2로 함.
        layer.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=2, shortcut=shortcut))
        for _ in range(1, num_blocks):
            layer.append(BasicBlock(in_channels=out_channels, out_channels=out_channels))   

        return nn.Sequential(*layer)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_pre(output)
        output = self.conv2_main(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)

        #output의 형태를 변화시킴. linear하게.
        output = output.view(output.size(0), -1)

        output = self.fc(output)
        
        return F.log_softmax(output)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256

## According to 3.4. Implementation ...
model = ResNet34().to(device=device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)



def train(epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 10 == 0:
            print('Train Epoch: %d | Batch Status: [%d/%d] \t| Loss: %.3f \t| Acc: %.3f%% (%d/%d)' % (epoch, batch_idx * len(data), len(trainloader.dataset), loss.item(), 100.0 * correct/total ,correct, total))

#https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    error1 = batch_size - correct[:1].reshape(-1).float().sum(0)
    error5 = batch_size - correct[:5].reshape(-1).float().sum(0)
    res = [error1, error5]
    return res

def test(epoch):
    model.eval()
    
    error_1_tot = 0
    error_5_tot = 0
    #don't need gradient calc
    with torch.no_grad():   
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            error_k = accuracy(output, target)
            error_1_tot += error_k[0]
            error_5_tot += error_k[1]
    #print("Test Epoch : %d" % (epoch))
    print("top-1 error : %.3f%% (%d/%d)" % (100.0 * error_1_tot/len(testloader.dataset), error_1_tot, len(testloader.dataset)))
    print("top-5 error : %.3f%% (%d/%d)" % (100.0 * error_5_tot/len(testloader.dataset), error_5_tot, len(testloader.dataset)))

if __name__ == "__main__":
    start = time.time()
    for epoch in range(100):
        train(epoch)
        print("Training Time %d min %d sec" % (int(time.time()-start)//60 , int(time.time()-start)%60))
        test(epoch)
        print("Training Time %d min %d sec" % (int(time.time()-start)//60 , int(time.time()-start)%60))
    print("End at %d min %d sec")
        
        



