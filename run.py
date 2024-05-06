import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score as fscore
import numpy as np
from utils import get_word_table, import_vecs, import_corpus
from container import TextCNN, LstmRNN, MLP
import wandb
import argparse

tblt = get_word_table()

def get_input(typex, batch_sz):
    # 导入input, assessments 可改参数path
    path_dir = ""
    if typex == 1:
        path_dir = "train.txt"
    elif typex == 2:
        path_dir = "validation.txt"
    elif typex == 3:
        path_dir = "test.txt"
    canvas_x, assessment = import_corpus(path_dir, tblt)
    dataset = TensorDataset(
        torch.from_numpy(canvas_x).type(torch.float), torch.from_numpy(assessment).type(torch.long)
    )
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_sz, num_workers=2)
    return dataloader


def train(dataloader):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    count, correct = 0, 0
    full_true = []
    full_pred = []

    for _, inputs in enumerate(dataloader):
        x = inputs[0]
        y = inputs[1]
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        # input sentences
        output = model(x)

        # input assessment
        loss = criterion(output, y)
        loss.backward()

        # optimize
        optimizer.step()
        train_loss += loss.item()

        # acc
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

        # ground truth
        full_true.extend(y.cpu().numpy().tolist())

        # predictions
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())

    # accessing
    train_loss *= batch_size
    train_loss /= len(dataloader.dataset)
    train_acc = correct / count
    scheduler.step()
    f1 = fscore(np.array(full_true), np.array(full_pred), average="binary")

    # save model?
    # torch.save(model.state_dict(), f"saving/{model.__name__}/my_cnn_model_{int(datetime.utcnow().timestamp())}.pth")
    return train_loss, train_acc, f1


def valid_and_test(dataloader):
    model.eval()
    cal_loss, cal_acc = 0.0, 0.0
    count, correct = 0, 0
    full_true = []
    full_pred = []
    for _, inputs in enumerate(dataloader):
        x = inputs[0]
        y = inputs[1]
        x, y = x.to(DEVICE), y.to(DEVICE)

        # input sentences
        output = model(x)

        # input assessment
        loss = criterion(output, y)
        cal_loss += loss.item()

        # no optimization

        # acc
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

        # ground truth
        full_true.extend(y.cpu().numpy().tolist())

        # predictions
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())

    # accessing
    cal_loss *= batch_size
    cal_loss /= len(dataloader.dataset)
    cal_acc = correct / count
    f1 = fscore(np.array(full_true), np.array(full_pred), average="binary")
    return cal_loss, cal_acc, f1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", "-e", dest="epoch", type=int, default=12)
    parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, default=32)
    parser.add_argument("--lr", "-l", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--network", "-n", dest="network", type=str, default="CNN")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ags = parse()
    epoch = ags.epoch
    learning_rate = ags.learning_rate
    batch_size = ags.batch_size
    train_dataloader = get_input(1, batch_size)
    test_dataloader = get_input(2, batch_size)
    valid_dataloader = get_input(3, batch_size)
    if ags.network == "CNN":
        print("choose cnn")
        model = TextCNN().to(DEVICE)
    elif ags.network == "RNN":
        print("choose rnn")
        model = LstmRNN().to(DEVICE)
    else:
        print("choose mlp")
        model = MLP().to(DEVICE)

    # 优化器 启动
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5)

    # 交叉熵评价
    criterion = nn.CrossEntropyLoss()

    # 训练
    wandb.init(project=f"Test", name=f"{model.__name__}_{learning_rate}", entity="Fl0at9973")
    wandb.config = {"learning_rate": learning_rate, "epochs": epoch, "batch_size": batch_size}
    for i in tqdm(range(1, epoch + 1, 1), "Training:"):
        train_loss, train_acc, train_f1 = train(train_dataloader)
        val_loss, val_acc, val_f1 = valid_and_test(valid_dataloader)
        test_loss, test_acc, test_f1 = valid_and_test(test_dataloader)

        # 画图
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
        })

        print(
            f"for epoch {i}, train_loss: {train_loss:.5f}, test_acc: {test_acc:.5f}"
        )