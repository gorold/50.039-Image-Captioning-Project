import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

sns.set()

from build_vocab import Vocabulary

if __name__ == '__main__':
    data_plots_save_path = os.path.join(os.getcwd(),'data_analysis_plots')
    if not os.path.exists(data_plots_save_path):
        os.makedirs(data_plots_save_path)

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    counts = []
    for word, count in vocab.word2count.items():
        counts.append(count)

    fig, ax = plt.subplots(1,1, figsize=(10,7))
    sns.distplot(counts, ax = ax, kde = False)
    ax.set_xlabel('Word Frequency', fontsize = 14)
    ax.set_ylabel('Number of Words', fontsize = 14)
    ax.set_title('Number of words per word frequency', fontsize = 14)
    plt.savefig(os.path.join(data_plots_save_path,'total_word_histogran.png'))
    plt.close()

    high_word_count = list(filter(lambda x:x[1] >= 100000, vocab.word2count.items()))
    print(high_word_count)

    low_word_count = list(filter(lambda x:x < 100, vocab.word2count.values()))
    fig, ax = plt.subplots(1,1, figsize=(10,7))
    sns.distplot(low_word_count, ax = ax, kde = False)
    ax.set_xlabel('Word Frequency')
    ax.set_ylabel('Number of Words')
    ax.set_title('Number of words per word frequency - for words with < 100 counts')
    plt.savefig(os.path.join(data_plots_save_path,'lpw_word_histogran.png'))
    plt.close()
