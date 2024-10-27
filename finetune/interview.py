import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pdb
from tqdm import tqdm

class Simple_Dataset(Dataset):
    def __init__ (self, data_num):
        self.data = torch.randn([data_num,3,2048, 2048])
        self.label = torch.randint(0, 10, (data_num, ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=4, padding=0, kernel_size=1)
        self.cv2 = nn.Conv2d(in_channels=64, out_channels=1, stride=4, padding=0, kernel_size=1)
        self.fc = nn.Linear(in_features=128*128, out_features=10)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x


def train(dataloader, model):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.01)

    for epoch in range(10):
        print(f"epoch: {epoch}")
        for (input, label) in tqdm(dataloader):
            opt.zero_grad()
            output = model(input)
            loss = crit(output, label)
            loss.backward()
            opt.step()

if __name__ == "__main__":
    simple_loader = DataLoader(Simple_Dataset(100), batch_size=1, shuffle=False)
    model = SimpleNet()
    train(simple_loader, model)