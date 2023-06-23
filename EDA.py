
import os
import pandas as pd
import pickle
from datetime import datetime
from tqdm.notebook import tqdm
from functools import lru_cache
import seaborn as sns
import matplotlib.pyplot as plt
def main():
    os.chdir('../dataset')
    abs_dir = os.getcwd()
    dataset_dir = abs_dir + '\\data'
    dir_list = os.listdir(dataset_dir) # ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    dir_dict = {}
    for l in dir_list: dir_dict[l] = os.listdir(f'{dataset_dir}/{l}')
    labels=['class', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL','source','filename']

    with open(os.path.join('./cached',f'df_onehot.pkl'), 'rb') as f:
        df_cut = pickle.load(f)
    print('Dataset loaded\n')
    sns.heatmap(df_cut.corr(numeric_only=True))
    plt.show()



if __name__ == '__main__':
    main()