import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime as dt
import sys

from tqdm import tqdm
from utils.misc import log_print

class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

class Epoch(object):
    def __init__(self, model, loss, stage_name, device='cpu', verbose=True, logger = None):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.logger = logger

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x):
        raise NotImplementedError

    def on_epoch_start(self):
        pass
    
    @staticmethod
    def correct_counts(y_pred, y):
        y_pred, y = y_pred.cpu(), y.cpu()
        mask = (y > -1)
        _, indices = torch.max(y_pred, 1)
        return (indices[mask] == y[mask]).float()

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        correct_count_meter = AverageValueMeter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            # Run for 1 epoch
            for x in iterator:
                loss, y_pred, y = self.batch_update(x)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'CrossEntropyLoss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                metric_value = self.correct_counts(y_pred, y).cpu().detach().numpy()
                correct_count_meter.add(metric_value.sum().item(),n = metric_value.shape[0])
                metrics_logs = {'Accuracy': correct_count_meter.mean}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        cumulative_logs = {'Accuracy': correct_count_meter.sum/correct_count_meter.n}
        cumulative_logs['loss'] = loss_meter.sum/loss_meter.n
        log_print(" ".join([f"{k}:{v:.4f}" for k, v in cumulative_logs.items()]), self.logger, log_only = True)

        return cumulative_logs

class TrainEpoch(Epoch):
    def __init__(self, model, loss, optimizer, device='cpu', verbose=True, logger = None):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='train',
            device=device,
            verbose=verbose,
            logger=logger
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x):
        self.optimizer.zero_grad()
        prediction, ground_truth = self.model.forward(x)
        loss = self.loss(prediction, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss, prediction, ground_truth

class ValidEpoch(Epoch):
    def __init__(self, model, loss, device='cpu', verbose=True, logger = None):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='valid',
            device=device,
            verbose=verbose,
            logger = logger
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x):
        with torch.no_grad():
            prediction, ground_truth = self.model.forward(x)
            loss = self.loss(prediction, ground_truth)
        return loss, prediction, ground_truth

def plot(train_losses, val_losses, train_acc, val_acc):
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].set_title('Loss Value')
    ax[0].plot(train_losses, color = 'skyblue', label="Training Loss")
    ax[0].plot(val_losses, color = 'orange', label = "Validation Loss")
    ax[0].legend()

    ax[1].set_title('Measure Value')
    ax[1].plot(train_acc, color = 'skyblue', label="Training Measure")
    ax[1].plot(val_acc, color = 'orange', label="Validation Measure")
    ax[1].legend()
    cwd = os.getcwd()

    if not os.path.exists(os.path.join(cwd,'plots')):
        os.makedirs(os.path.join(cwd,'plots'))
    plt.savefig(os.path.join(cwd,'plots','nn_training' + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    plt.close()

def train_model(train_dataloader,
                validation_dataloader,
                model,
                loss,
                optimizer,
                scheduler = None,
                batch_size = 1,
                num_epochs = 10,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'weights'),
                model_save_prefix = '',
                plots_save_path = os.path.join(os.getcwd(),'plots')
                ):

    if torch.cuda.is_available():
        log_print('Using GPU', logger)
    else:
        log_print('Using CPU', logger)
    
    # Define Epochs
    train_epoch = TrainEpoch(
        model = model,
        loss = loss, 
        optimizer = optimizer,
        device = device,
        verbose = verbose,
        logger = logger,
    )

    valid_epoch = ValidEpoch(
        model = model, 
        loss = loss, 
        device = device,
        verbose = verbose,
        logger = logger,
    )

    # Record for plotting
    losses = {'train':[],'val':[]}
    metric_values = {'train':{'Accuracy':[]},'val':{'Accuracy':[]}}

    # Run Epochs
    best_perfmeasure = 0
    best_epoch = -1
    start_time = dt.datetime.now()
    log_print('Training model...', logger)

    for epoch in range(num_epochs):
        log_print(f'\nEpoch: {epoch}', logger)

        train_logs = train_epoch.run(train_dataloader)
        losses['train'].append(train_logs['loss'])
        metric_values['train']['Accuracy'].append(train_logs['Accuracy'])

        valid_logs = valid_epoch.run(validation_dataloader)
        losses['val'].append(valid_logs['loss'])
        metric_values['val']['Accuracy'].append(valid_logs['Accuracy'])
        
        log_print('Random Sampling', logger)
        for _ in range(20):
            log_print(model.sample_caption(torch.tensor([0])), logger)
        log_print('Beam Search', logger)
        log_print(model.beam_search(torch.tensor([0])), logger)

        if scheduler is not None:
            scheduler.step()
            log_print(f"Next Epoch Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}", logger)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if best_perfmeasure < valid_logs['Accuracy']: # Right now the metric to be chosen for best_perf_measure is always the first metric
            best_perfmeasure = valid_logs['Accuracy']
            best_epoch = epoch

            torch.save(model, os.path.join(model_save_path,model_save_prefix + 'best_model.pth'))
            log_print('Best Model Saved', logger)

        torch.save(model, os.path.join(model_save_path,model_save_prefix + 'current_model.pth'))
        log_print('Current Model Saved', logger)

    log_print(f'Best epoch: {best_epoch} Best Performance Measure: {best_perfmeasure:.5f}', logger)
    log_print(f'Time Taken to train: {dt.datetime.now()-start_time}', logger)

    # Implement plotting feature
    plot(losses['train'],losses['val'],metric_values['train']['Accuracy'],metric_values['val']['Accuracy'])
    log_print('Plot Saved', logger)