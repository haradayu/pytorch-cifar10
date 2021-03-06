import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse

from models import *
from misc import progress_bar

import time


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Parameter:
    def __init__(self):
        self.lr = None
        self.epoch = 1000 
        self.trainBatchSize = None
        self.testBatchSize = 100
        self.cuda = True
        self.learningTime = 120

def objective(trial):
    # parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    # parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    # parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    # parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    # parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    # parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    # args = parser.parse_args()

    parameter = Parameter()
    parameter.lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064,0.128, 0.256, 0.512])
    parameter.trainBatchSize = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    solver = Solver(parameter, trial)
    
    accuracy, total_time = solver.run()
    return (1 - accuracy)

def main():
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner
    study = optuna.create_study(pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))

class Solver(object):
    def __init__(self, config, trial):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.leaning_time = config.learningTime

        self.trial = trial
        print("lr:", self.lr, "batchsize:",self.train_batch_size)

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = LeNet().to(self.device)
        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        self.model = WideResNet(depth=28, num_classes=10).to(self.device)
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model) 

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self, start_time):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            if time.time() - start_time > self.leaning_time:
                break

            # progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                        #  % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                # progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                            #  % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        from pathlib import Path
        Path("../model/").mkdir(exist_ok=True)
        model_out_path = "../model/model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        start = time.time()
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            train_result = self.train(start)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            print(f"{time.time() - start}, {epoch}, {train_result[0]}, {train_result[1]}, {test_result[0]}, {test_result[1]}")
            # if epoch == self.epochs:

            if time.time() - start > self.leaning_time:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                break
                # self.save()
            if self.trial.should_prune(epoch):
                raise optuna.structs.TrialPruned()
        return accuracy, time.time() - start


if __name__ == '__main__':
    main()
