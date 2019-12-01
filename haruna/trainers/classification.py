import os
import tempfile

import mlconfig
import mlflow
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from ..metrics import Accuracy, Average
from .trainer import Trainer


class ImageClassificationTrainer(Trainer):

    def __init__(self, device, model, optimizer, scheduler, train_loader, valid_loader, num_epochs):
        super(ImageClassificationTrainer, self).__init__()
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs

        self.register_status('epoch', 1)
        self.register_status('best_acc', 0)
        self.temp_dir = tempfile.gettempdir()

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate()
            self.scheduler.step()

            self.save_checkpoint(os.path.join(self.temp_dir, 'checkpoint.pth'))

            metrics = dict(train_loss=train_loss.value,
                           train_acc=train_acc.value,
                           valid_loss=valid_loss.value,
                           valid_acc=valid_acc.value)
            mlflow.log_metrics(metrics, step=self.epoch)

            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {}, train acc: {}, '.format(train_loss, train_acc)
            format_string += 'valid loss: {}, valid acc: {}, '.format(valid_loss, valid_acc)
            format_string += 'best valid acc: {}.'.format(self.best_acc)
            tqdm.write(format_string)

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        eval_loss = Average()
        eval_acc = Accuracy()

        for x, y in tqdm(self.valid_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            eval_loss.update(loss.item(), number=x.size(0))
            eval_acc.update(output, y)

        if eval_acc > self.best_acc:
            self.best_acc = eval_acc
            self.save_model(os.path.join(self.temp_dir, 'best.pth'))

        return eval_loss, eval_acc

    def save_model(self, f):
        torch.save(self.model.state_dict(), f)
        mlflow.log_artifact(f)

    def save_checkpoint(self, f):
        state_dict = self.state_dict()
        torch.save(state_dict, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        state_dict = torch.load(f, map_location=self.device)
        self.load_state_dict(state_dict)
        self.epoch += 1


@mlconfig.register
def train_image_classification(config, device, num_epochs):
    model = config.model()
    model.to(device)
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    train_loader = config.dataset(train=True)
    valid_loader = config.dataset(train=False)

    trainer = ImageClassificationTrainer(device, model, optimizer, scheduler, train_loader, valid_loader, num_epochs)

    return trainer
