import argparse
import os
import time

import tinynn as tn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Dense(nn.Module):

    def __init__(self):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 70)
        self.fc4 = nn.Linear(70, 30)
        self.fc5 = nn.Linear(30, 10)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        return x


class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding="same")
        self.conv2 = nn.Conv2d(6, 16, 5, 1, padding="same")

        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        x = F.log_softmax(x, dim=1)
        return x


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.recurrent = nn.RNN(28, 30, batch_first=True)
        self.fc1 = nn.Linear(30, 10)

    def forward(self, x):
        output, hidden = self.recurrent(x)
        x = output[:, -1]
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


class LSTM(RNN):

    def __init__(self):
        super(LSTM, self).__init__()
        self.recurrent = nn.LSTM(28, 30, batch_first=True)
        self.fc1 = nn.Linear(30, 10)


def main():
    if args.seed >= 0:
        tn.seeder.random_seed(args.seed)
        torch.manual_seed(args.seed)

    mnist = tn.dataset.MNIST(args.data_dir, one_hot=False)
    train_x, train_y = mnist.train_set
    test_x, test_y = mnist.test_set

    if args.model_type == "mlp":
        model = Dense()
    elif args.model_type == "cnn":
        train_x = train_x.reshape((-1, 1, 28, 28))
        test_x = test_x.reshape((-1, 1, 28, 28))
        model = Conv()
    elif args.model_type == "rnn":
        train_x = train_x.reshape((-1, 28, 28))
        test_x = test_x.reshape((-1, 28, 28))
        model = RNN()
    elif args.model_type == "lstm":
        train_x = train_x.reshape((-1, 28, 28))
        test_x = test_x.reshape((-1, 28, 28))
        model = LSTM()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    iterator = tn.data_iterator.BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        f_cost, b_cost = 0, 0
        for batch in iterator(train_x, train_y):
            x = torch.from_numpy(batch.inputs).to(device)
            y = torch.from_numpy(batch.targets).to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.nll_loss(pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} time cost: {time.time() - t_start}")
        # evaluate
        evaluate(model, test_x, test_y)


def evaluate(model, test_x, test_y):
    model.eval()
    x, y = torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device)
    with torch.no_grad():
        pred = model(x)
        test_pred_idx = pred.argmax(dim=1).numpy()
        accuracy, info = tn.metric.accuracy(test_pred_idx, test_y)
        print(f"accuracy: {accuracy:.4f} info: {info}")


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--model_type", default="mlp", type=str,
                        help="[*mlp|cnn|rnn|lstm]")
    parser.add_argument("--num_ep", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=31, type=int)
    args = parser.parse_args()

    device = torch.device("cpu")

    main()
