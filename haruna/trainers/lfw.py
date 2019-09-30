import os
import tempfile

import mlconfig
import mlflow
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm, trange

from ..metrics import Accuracy, Average
from .trainer import AbstractTrainer


@mlconfig.register
class LFWTrainer(AbstractTrainer):

    def __init__(self, config, device, num_epochs):
        model = config.model()
        model.to(device)
        optimizer = config.optimizer(model.parameters())
        scheduler = config.scheduler(optimizer)
        transform = config.transform()
        train_loader = config.dataset(train=True, transform=transform)
        valid_loader = config.dataset(train=False, transform=transform)

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs

        self.epoch = 1
        self.best_acc = 0
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

        for image, target in tqdm(self.train_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            output = self.model(image)['out']
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=image.size(0))
            train_acc.update(output, target)

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            for image, target in tqdm(self.valid_loader):
                image = image.to(self.device)
                target = target.to(self.device)

                output = self.model(image)['out']
                loss = F.cross_entropy(output, target)

                valid_loss.update(loss.item(), number=image.size(0))
                valid_acc.update(output, target)

        if valid_acc > self.best_acc:
            self.best_acc = valid_acc
            self.save_model(os.path.join(self.temp_dir, 'best.pth'))

        self._save_image(image, output)

        return valid_loss, valid_acc

    def save_model(self, f):
        self.model.eval()

        torch.save(self.model.state_dict(), f)
        mlflow.log_artifact(f)

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']

    def _save_image(self, x, y, alpha=0.5):
        image_path = os.path.join(self.temp_dir, 'epoch_{:04d}.jpg'.format(self.epoch))

        with torch.no_grad():
            x = (x - x.min()) / (x.max() - x.min())
            y = torch.softmax(y, dim=1).ge(0.5).float()
            z = (1 - alpha) * x + alpha * y

            image = torch.cat([x, y, z]).detach().cpu()

        save_image(image, image_path, nrow=x.size(0))
        mlflow.log_artifact(image_path)
