import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime as dt
import sys
import json

from tqdm import tqdm
from copy import deepcopy
from util.metrics import caption_list_to_words
from PIL import Image

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
                encoder_unfreeze_epoch = 3,
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

    encoder_frozen = True
    for epoch in range(num_epochs):
        # if encoder_frozen and epoch == encoder_unfreeze_epoch:
        #     for p in model[0].parameters():
        #         p.requires_grad = True
        #     encoder_frozen = False
        #     log_print(f'Encoder Unfrozen', logger)
        
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
                    coco,
                    validation_path,
                    encoder,
                    decoder,
                    metrics,
                    metrics_to_plot = ['Accuracy','Bleu_1','Bleu_2','Bleu_3','Bleu_4','Rouge'],
                    top_n = 2,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    logger = None,
                    verbose = True,
                    plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
                    prefix = 'Val'
                    ):
    # Plots the top N images in terms of Bleu Scores scores
    def _format_string(blue_rouge):
        format_bleu_rouge = ''
        for key, v in bleu_rouge.items():
            format_bleu_rouge += f'{key}: {v:.2e}\n'
        return format_bleu_rouge

    # Helper function to plot mask and original pictures
    def _plot_topn(start_rank, rank_split, rank_processed, metrics_to_plot, low_high_str):
        cols = len(metrics_to_plot)
        fig, ax = plt.subplots(rank_split,cols, figsize=(6*cols, 4*rank_split))
        for i in range(rank_split):
            for m_idx, metric in enumerate(metrics_to_plot):
                # Plot the Ground Truth
                ax[i,m_idx].imshow(rank_processed[metric][start_rank+i][0])
                ax[i,m_idx].set_xlabel(f'GT: {rank_processed[metric][start_rank+i][2]}\nPredicted: {rank_processed[metric][start_rank+i][1]}')
                ax[i,m_idx].set_title(f'{metric}:{rank_processed[metric][start_rank+i][3]:.3f}', fontsize=12)
                
        current_time = str(dt.datetime.now())[0:10].replace('-','_')
        if not os.path.exists(os.path.join(plots_save_path,current_time)):
            os.makedirs(os.path.join(plots_save_path,current_time))
        fig.suptitle(f'{low_high_str} {start_rank} to {start_rank+rank_split}', fontsize=20)
        fig.tight_layout(pad = 5.0)
        plt.savefig(os.path.join(plots_save_path,current_time, f"{prefix}_{low_high_str}_{start_rank+rank_split}.png"))
        plt.close()
        log_print(f'Plot {low_high_str} {start_rank+rank_split} saved', logger)

    log_print('Running Inference...', logger)
    picture_scores = {}

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    with tqdm(validation_dataloader, desc='Inference', file=sys.stdout, disable=False) as iterator:
        for _, (x, y, lengths, img_ids) in enumerate(iterator):
            assert x.shape[0] == 1, 'Batch size must be 1'
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = encoder(x)
                y_pred, _ = decoder.beam_search(features) # Note y_pred is a list
            y_pred = y_pred[1:-1]
            y = y[:,1:-1] # Offset the tensor to exclude start token for the calculation of metrics
            accuracy = metrics[0].evaluate([y_pred], y.tolist()) # Compute accuracy
            bleu_rouge_scores = metrics[1].evaluate([y_pred], y, img_ids)
            
            bleu_rouge = {'Accuracy':accuracy,
                            'Bleu_1': bleu_rouge_scores[0],
                            'Bleu_2': bleu_rouge_scores[1],
                            'Bleu_3': bleu_rouge_scores[2],
                            'Bleu_4': bleu_rouge_scores[3],
                            'Rouge': bleu_rouge_scores[4]}

            picture_scores[img_ids[0]] = deepcopy(bleu_rouge), caption_list_to_words([y_pred],decoder.vocab)[0], caption_list_to_words(y.tolist(),decoder.vocab)[0]

    lowest = {}
    highest = {}
    for metric in metrics_to_plot:
        sorted_scores = sorted(picture_scores.items(), key = lambda x:x[1][0][metric])
        sorted_scores = [(idx, value) for idx, value in enumerate(sorted_scores)]
        lowest[metric] = sorted_scores[:top_n]
        lowest[metric] = {value[1][0]:(value[0],value[1][1][0][metric],value[1][1][1],value[1][1][2]) for value in lowest[metric]}
        highest[metric] = sorted_scores[-1*top_n:]
        highest[metric] = {value[1][0]:(top_n-idx-1,value[1][1][0][metric],value[1][1][1],value[1][1][2]) for idx, value in enumerate(highest[metric])}

    lowest_processed = {metric:[0 for _ in range(top_n)] for metric in metrics_to_plot}
    highest_processed = {metric:[0 for _ in range(top_n)] for metric in metrics_to_plot}

    for metric in metrics_to_plot:
        for image_id in lowest[metric].keys():
            path = coco.loadImgs(image_id)[0]['file_name']
            img = np.array(Image.open(os.path.join(validation_path, path)).convert('RGB'))
            rank, metric_value, predicted_caption, gt_caption = lowest[metric][image_id]
            lowest_processed[metric][rank] = (img, predicted_caption, gt_caption, metric_value)
        
        for image_id in highest[metric].keys():
            path = coco.loadImgs(image_id)[0]['file_name']
            img = np.array(Image.open(os.path.join(validation_path, path)).convert('RGB'))
            rank, metric_value, predicted_caption, gt_caption = highest[metric][image_id]
            highest_processed[metric][rank] = (img, predicted_caption, gt_caption, metric_value)

    rank_split = 5
    for plot_idx in range(top_n//rank_split):
        start_rank = plot_idx*rank_split
        end_rank = min(plot_idx*rank_split+rank_split,top_n)
        _plot_topn(start_rank, end_rank-start_rank,lowest_processed, metrics_to_plot, 'Lowest')
        _plot_topn(start_rank, end_rank-start_rank,highest_processed, metrics_to_plot, 'Highest')

    log_print(f'Validation completed', logger)

def greedy_prediction(validation_dataloader,
                    encoder,
                    decoder,
                    metrics,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    logger = None,
                    validate = True,
                    results_save_path = os.path.join(os.getcwd(),'results')):

    log_print('Running Inference...', logger)
    results = []

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    y_preds = []
    image_id_list = []
    with tqdm(validation_dataloader, desc='Inference', file=sys.stdout, disable=False) as iterator:
        for _, (x, y, _, img_ids) in enumerate(iterator):
            x = x.to(device)
            with torch.no_grad():
                features = encoder(x)
                y_pred = decoder.sample(features) # Note y_pred is a list

            y_pred = [tnsr[:-1].tolist() for tnsr in y_pred]
            y_preds += y_pred
            image_id_list += img_ids
            
            for image_id, caption in zip(img_ids, caption_list_to_words(y_pred,decoder.vocab)):
                results.append({'image_id':image_id,'caption':caption})

    if validate:
        bleu_rouge_scores = metrics[1].evaluate(y_preds, None, image_id_list)
        bleu_rouge = {'Bleu_1': bleu_rouge_scores[0],
                        'Bleu_2': bleu_rouge_scores[1],
                        'Bleu_3': bleu_rouge_scores[2],
                        'Bleu_4': bleu_rouge_scores[3],
                        'Rouge': bleu_rouge_scores[4]}
        for k, v in bleu_rouge.items():
            print(f'{k}:{v:.5f}')

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    
    with open(os.path.join(results_save_path,'results.json'), 'w') as outfile:
        json.dump(results, outfile)