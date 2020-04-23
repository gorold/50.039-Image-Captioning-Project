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
                input_lengths = [l-1 for l in lengths]
                loss, y_pred = self.batch_update(x, y, input_lengths)
                y = y[:,1:] # Offset the tensor to exclude start token for the calculation of metrics
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
        loss = self.loss(predictions, captions[:,1:])
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
            loss = self.loss(predictions, captions[:,1:])
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

def validate_and_plot(validation_dataloader,
                    validation_dataloader_org,
                    model,
                    metrics,
                    top_n = 20,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    logger = None,
                    verbose = True,
                    plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
                    prefix = 'Val',
                    threshold = 0.5
                    ):
    # Plots the top N images in terms of Bleu Scores scores

    # Helper function to return the image, the predicted mask and the actual mask for any datapoint
    def _get_image_mask(data, model, class_idx):
        aug_data, org_data = data
        inputs = aug_data[0].to(device)
        masks_pred = model(inputs)

        img = org_data[0][0].detach().cpu()
        mask = aug_data[1][0][class_idx].cpu()
        mask_pred = masks_pred[0][class_idx].cpu()
        mask_pred = np.where(mask_pred > threshold,1,0)
        trans = torchvision.transforms.ToPILImage()
        
        img = np.array(trans(img))
        return img, mask_pred, mask

    # Helper function to plot mask and original pictures
    def _plot_topn(rank, metric_info, low_high_str):
        fig, ax = plt.subplots(2*len(metrics),len(classes), figsize=(7*len(classes), 10*len(metrics)))
        for class_idx, _class in enumerate(classes):
            for m_idx, metric in enumerate(metrics):
                # Plot the Ground Truth
                if len(classes) > 1:
                    ax[m_idx*2,class_idx].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2,class_idx].imshow(metric_info[_class,metric.__name__][rank][2], alpha=0.3, cmap='gray')
                    ax[m_idx*2,class_idx].set_title(f'{_class} Ground Truth {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)

                    # Plot the picture and the masks
                    ax[m_idx*2+1,class_idx].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2+1,class_idx].imshow(metric_info[_class,metric.__name__][rank][1], alpha=0.3, cmap='gray')
                    ax[m_idx*2+1,class_idx].set_title(f'{_class} Prediction {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)
                else:
                    ax[m_idx*2].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2].imshow(metric_info[_class,metric.__name__][rank][2], alpha=0.3, cmap='gray')
                    ax[m_idx*2].set_title(f'{_class} Ground Truth {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)

                    # Plot the picture and the masks
                    ax[m_idx*2+1].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2+1].imshow(metric_info[_class,metric.__name__][rank][1], alpha=0.3, cmap='gray')
                    ax[m_idx*2+1].set_title(f'{_class} Prediction {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)
                
        current_time = str(dt.datetime.now())[0:10].replace('-','_')
        if not os.path.exists(os.path.join(plots_save_path,current_time)):
            os.makedirs(os.path.join(plots_save_path,current_time))
        fig.suptitle(f'{low_high_str} {rank+1}', fontsize=20)
        plt.savefig(os.path.join(plots_save_path,current_time, f"{prefix}_{low_high_str}_{rank+1}.png"))
        plt.close()
        log_print(f'Plot {low_high_str} {rank+1} saved', logger)

    log_print('Running Inference...', logger)
    picture_iou_scores = OrderedDict()

    if torch.cuda.is_available():
        model.cuda()

    with tqdm(validation_dataloader, desc='Inference Round 1', file=sys.stdout, disable=False) as iterator:
        for data_idx, data in enumerate(iterator):
            inputs = data[0].to(device)
            masks_pred = model(inputs)

            # Make sure batch size is 1
            assert len(inputs) == 1
            masks = data[1][0].to(device)
            masks_pred = masks_pred[0]
            if masks_pred.shape[0] != len(classes):
                raise Exception('Your model predicts more classes than the number of classes specified')

            for class_idx in range(len(classes)):
                for metric in metrics:
                    picture_iou_scores[data_idx,classes[class_idx],metric.__name__] = float(metric(masks_pred[class_idx,:,:], masks[class_idx,:,:]).cpu().detach().numpy())
    
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

    # Only do it for the first metric
    lowest = {}
    highest = {}
    for _class in classes:
        for metric in metrics:
            sorted_iou_scores = sorted(filter(lambda x:x[0][1] == _class and x[0][2] == metric.__name__, picture_iou_scores.items()), key = lambda x:x[1])
            sorted_iou_scores = [(idx, value[0][0], value[1]) for idx, value in enumerate(sorted_iou_scores)]
            lowest[_class,metric.__name__] = sorted_iou_scores[:top_n]
            lowest[_class,metric.__name__] = {value[1]:(value[0],value[2]) for value in lowest[_class,metric.__name__]}
            highest[_class,metric.__name__] = sorted_iou_scores[-1*top_n:]
            highest[_class,metric.__name__] = {value[1]:(idx,value[2]) for idx, value in enumerate(highest[_class,metric.__name__])}

    lowest_processed = {(_class, metric.__name__):[0 for _ in range(top_n)] for _class in classes for metric in metrics}
    highest_processed = {(_class, metric.__name__):[0 for _ in range(top_n)] for _class in classes for metric in metrics}

    with tqdm(zip(validation_dataloader,validation_dataloader_org), desc='Inference Round 2', file=sys.stdout, disable=False) as iterator:
        for data_idx, data in enumerate(iterator):
            for class_idx, _class in enumerate(classes):
                for metric in metrics:
                    if data_idx in lowest[_class,metric.__name__].keys():
                        img, mask_pred, mask = _get_image_mask(data, model, class_idx)
                        rank = lowest[_class,metric.__name__][data_idx][0]
                        metric_value = lowest[_class,metric.__name__][data_idx][1]
                        lowest_processed[_class,metric.__name__][rank] = (img, mask_pred, mask, metric_value)
                    
                    elif data_idx in highest[_class,metric.__name__].keys():
                        img, mask_pred, mask = _get_image_mask(data, model, class_idx)
                        rank = highest[_class,metric.__name__][data_idx][0]
                        metric_value = highest[_class,metric.__name__][data_idx][1]
                        highest_processed[_class,metric.__name__][rank] = (img, mask_pred, mask, metric_value)

    for rank in range(top_n):
        _plot_topn(rank, lowest_processed, 'Lowest')
        _plot_topn(rank, highest_processed, 'Highest')

    log_print(f'Validation completed', logger)