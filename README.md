# cs229-project

## Progress
### Initial Model and Hyperparams:
```
batch_size = 64
learning_rate = 0.001
num_epochs = 200

class Model(nn.Module):
    def __init__(self, num_classes=30):
        super(Model, self).__init__()
        # input = 224 x 224 x 3
        self.conv1 = nn.Conv2d(3, 10, 5)          # 220 x 220 x 10
        self.pool1 = nn.MaxPool2d(2, 2)           # 110 x 110 x 10
        self.conv2 = nn.Conv2d(10, 20, 5)         # 106 x 106 x 20
        self.pool2 = nn.MaxPool2d(2, 2)           #  53 x  53 x 20
        self.conv3 = nn.Conv2d(20, 32, 3)         #  51 x  51 x 32
        self.pool3 = nn.MaxPool2d(2, 2)           #  25 x  25 x 32
        self.fc1 = nn.Linear(25 * 25 * 32, 30)    #        30

    def forward(self, x):
        # run your data through the layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 25 * 25 * 32)
        x = self.fc1(x).squeeze()
        return x
```
#### Results:
```
Initial Evaluation: 
Accuracy of the network on the 45600 train images: 3.267543859649123%
Accuracy of the network on the 5700 val images: 3.210526315789474%
==================================================
Accuracy of the network on the 45600 train images: 14.432017543859649%
Accuracy of the network on the 5700 val images: 11.68421052631579%
==================================================
Accuracy of the network on the 45600 train images: 23.662280701754387%
Accuracy of the network on the 5700 val images: 14.017543859649123%
==================================================
Accuracy of the network on the 45600 train images: 39.375%
Accuracy of the network on the 5700 val images: 14.789473684210526%
==================================================
Accuracy of the network on the 45600 train images: 57.66447368421053%
Accuracy of the network on the 5700 val images: 14.56140350877193%
==================================================
Accuracy of the network on the 45600 train images: 74.0657894736842%
Accuracy of the network on the 5700 val images: 13.070175438596491%
```
### Updated Model 1 and Hyperparams:
```
batch_size = 64
learning_rate = 0.001
num_epochs = 200

reg = "l1"
lambda_l1 = 0.0001
lambda_l2 = 0.001

aug = torchvision.transforms.Compose([
      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```
#### Results:
```
Initial Evaluation: 
Accuracy of the network on the 45600 train images: 3.2149122807017543%
Accuracy of the network on the 5700 val images: 2.9649122807017543%
==================================================
Accuracy of the network on the 45600 train images: 5.078947368421052%
Accuracy of the network on the 5700 val images: 5.280701754385965%
==================================================
Accuracy of the network on the 45600 train images: 5.212719298245614%
Accuracy of the network on the 5700 val images: 4.87719298245614%
==================================================
Accuracy of the network on the 45600 train images: 6.833333333333333%
Accuracy of the network on the 5700 val images: 6.157894736842105%
==================================================
Accuracy of the network on the 45600 train images: 9.25%
Accuracy of the network on the 5700 val images: 7.491228070175438%
==================================================
Accuracy of the network on the 45600 train images: 13.875%
Accuracy of the network on the 5700 val images: 10.929824561403509%
==================================================
```
