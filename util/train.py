import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime as dt
import sys

from tqdm import tqdm

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)

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
    def __init__(self, encoder, decoder, loss, metrics, stage_name, device='cpu', verbose=True, logger = None):
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.logger = logger

        self._to_device()

    def _to_device(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, *args):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        correct_count_meter = AverageValueMeter()
        bleu_rouge = [AverageValueMeter() for i in range(5)]

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            # Run for 1 epoch
            for x, y, lengths, img_ids in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y, lengths)

                # update loss logs
                loss_value = loss.item()
                loss_meter.add(loss_value)
                loss_logs = {'CrossEntropyLoss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                metric_value = self.metrics[0].evaluate(y_pred, y)
                correct_count_meter.add(metric_value.sum().item(),n = metric_value.shape[0])

                # Update Bleu Rouge Scores
                metric_values = self.metrics[1].evaluate(y_pred, y, img_ids)
                for i in range(5):
                    bleu_rouge[i].add(metric_values[i])

                metrics_logs = {'Accuracy': correct_count_meter.mean,
                                'Bleu_1': bleu_rouge[0].mean,
                                'Bleu_2': bleu_rouge[1].mean,
                                'Bleu_3': bleu_rouge[2].mean,
                                'Bleu_4': bleu_rouge[3].mean,
                                'Rouge': bleu_rouge[4].mean}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        cumulative_logs = {'accuracy': correct_count_meter.sum/correct_count_meter.n,
                            'bleu_1': bleu_rouge[0].sum/bleu_rouge[0].n,
                            'bleu_2': bleu_rouge[1].sum/bleu_rouge[1].n,
                            'bleu_3': bleu_rouge[2].sum/bleu_rouge[2].n,
                            'bleu_4': bleu_rouge[3].sum/bleu_rouge[3].n,
                            'rouge': bleu_rouge[4].sum/bleu_rouge[4].n}
        cumulative_logs['loss'] = loss_meter.sum/loss_meter.n
        log_print(" ".join([f"{k}:{v:.4f}" for k, v in cumulative_logs.items()]), self.logger, log_only = True)

        return cumulative_logs

class TrainEpoch(Epoch):
    def __init__(self, encoder, decoder, loss, metrics, optimizer, device='cpu', verbose=True, logger = None):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss=loss,
            metrics = metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            logger=logger
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.encoder.train()
        self.decoder.train()

    def batch_update(self, images, captions, lengths):
        self.optimizer.zero_grad()

        features = self.encoder(images) # The encoder generates the features, which is passed into the LSTM as the first input
        predictions, _ = self.decoder(features, captions, lengths)
        loss = self.loss(predictions, captions)
        loss.backward()
        self.optimizer.step()

        return loss, predictions

class ValidEpoch(Epoch):
    def __init__(self, encoder, decoder, loss, metrics, device='cpu', verbose=True, logger = None):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss=loss,
            metrics = metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            logger=logger
        )

    def on_epoch_start(self):
        self.encoder.eval()
        self.decoder.eval()

    def batch_update(self, images, captions, lengths):
        with torch.no_grad():
            features = self.encoder(images)
            predictions, _ = self.decoder(features, captions, lengths)
            loss = self.loss(predictions, captions)
        return loss, predictions

def plot(train_losses, val_losses, train_metrics, val_metrics):
    fig, ax = plt.subplots(1,4, figsize = (20,5))
    ax[0].set_title('Loss Value')
    ax[0].plot(train_losses, color = 'skyblue', label="Training Loss")
    ax[0].plot(val_losses, color = 'orange', label = "Validation Loss")
    ax[0].legend()

    ax[1].set_title('Accuracy')
    ax[1].plot(train_metrics['accuracy'], color = 'skyblue', label="Training Accuracy")
    ax[1].plot(val_metrics['accuracy'], color = 'orange', label="Validation Accuracy")
    ax[1].legend()

    ax[2].set_title('Bleu')
    ax[2].plot(train_metrics['bleu_1'], color = 'skyblue', label="Training Bleu 1")
    ax[2].plot(train_metrics['bleu_2'], color = 'dodgerblue', label="Training Blue 2")
    ax[2].plot(train_metrics['bleu_3'], color = 'royalblue', label="Training Bleu 3")
    ax[2].plot(train_metrics['bleu_4'], color = 'navy', label="Training Bleu 4")
    ax[2].plot(val_metrics['bleu_1'], color = 'lightcoral', label="Validation Bleu 1")
    ax[2].plot(val_metrics['bleu_2'], color = 'indianred', label="Validation Blue 2")
    ax[2].plot(val_metrics['bleu_3'], color = 'brown', label="Validation Bleu 3")
    ax[2].plot(val_metrics['bleu_4'], color = 'maroon', label="Validation Bleu 4")
    ax[2].legend()

    ax[3].set_title('Rouge')
    ax[3].plot(train_metrics['rouge'], color = 'skyblue', label="Training Rouge")
    ax[3].plot(val_metrics['rouge'], color = 'orange', label="Validation Rouge")
    ax[3].legend()
    cwd = os.getcwd()

    if not os.path.exists(os.path.join(cwd,'plots')):
        os.makedirs(os.path.join(cwd,'plots'))
    plt.savefig(os.path.join(cwd,'plots','nn_training' + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    plt.close()

def train_model(train_dataloader,
                validation_dataloader,
                model,
                loss,
                train_metrics,
                val_metrics,
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
        encoder = model[0],
        decoder = model[1],
        loss = loss,
        metrics = train_metrics,
        optimizer = optimizer,
        device = device,
        verbose = verbose,
        logger = logger,
    )

    valid_epoch = ValidEpoch(
        encoder = model[0],
        decoder = model[1],
        loss = loss,
        metrics = val_metrics,
        device = device,
        verbose = verbose,
        logger = logger,
    )

    # Record for plotting
    metric_names = ['accuracy','bleu_1','bleu_2','bleu_3','bleu_4','rouge']
    losses = {'train':[],'val':[]}
    metric_values = {'train':{name:[] for name in metric_names},'val':{name:[] for name in metric_names}}

    # Run Epochs
    best_perfmeasure = 0
    best_epoch = -1
    start_time = dt.datetime.now()
    log_print('Training model...', logger)

    for epoch in range(num_epochs):
        log_print(f'\nEpoch: {epoch}', logger)

        train_logs = train_epoch.run(train_dataloader)
        losses['train'].append(train_logs['loss'])
        for metric in metric_names:
            metric_values['train'][metric].append(train_logs[metric])

        valid_logs = valid_epoch.run(validation_dataloader)
        losses['val'].append(valid_logs['loss'])
        for metric in metric_names:
            metric_values['val'][metric].append(valid_logs[metric])

        if scheduler is not None:
            scheduler.step()
            log_print(f"Next Epoch Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}", logger)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if best_perfmeasure < valid_logs['accuracy']: # Right now the metric to be chosen for best_perf_measure is always the first metric
            best_perfmeasure = valid_logs['accuracy']
            best_epoch = epoch

            torch.save(model, os.path.join(model_save_path,model_save_prefix + 'best_model.pth'))
            log_print('Best Model Saved', logger)

        torch.save(model, os.path.join(model_save_path,model_save_prefix + 'current_model.pth'))
        log_print('Current Model Saved', logger)

    log_print(f'Best epoch: {best_epoch} Best Performance Measure: {best_perfmeasure:.5f}', logger)
    log_print(f'Time Taken to train: {dt.datetime.now()-start_time}', logger)

    # Implement plotting feature
    plot(losses['train'],losses['val'],metric_values['train'],metric_values['val'])
    log_print('Plot Saved', logger)