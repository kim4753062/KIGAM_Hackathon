from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import seaborn as sns
from copy import copy
from math import *
'''
Utility ################################################################################################################
make_scaler: dataframe에서 주어진 id set의 변수값을 scaling하는 scaler를 반환
add_noise: 데이터(batch)에서 random noise를 첨가 => vicinal risk minimization # Advances in Neural Information Processing Systems 13 (NIPS 2000)
random_masking: 0에서 1 사이의 랜덤값이 mask ratio을 넘어설 때, 배치 마스킹 비율만큼의 배치에서 하나의 변수를 0으로 masking함
use_synthetic: 학습에서 simulated data와 drawn data를 사용할 경우, 각 data마다 scaler를 반환
onehot_to_label: 학습 결과로 도출된 출력을 label로 변환하여 반환 
########################################################################################################################
'''
def make_scaler(id_set, df, vars):
    df = copy(df)
    idx=False
    for id in id_set:
        idx +=df['instance_id'] == id
    only = df[idx][vars]
    scaler = StandardScaler()
    scaler.fit(only)
    return scaler

def fit_group_scaler(id_set, df, vars):
    df = copy(df)
    idx=False
    for id in id_set:
        idx +=df['instance_id'] == id
    only = df[idx][vars]
    scaler = MinMaxScaler()
    scaler.fit(only)
    df.loc[idx, vars] = scaler.fit_transform(only)
    return df

def make_stats_scaler(id_set, df, vars):
    df = copy(df)
    idx=False
    for id in id_set:
        idx +=df['instance_id'] == id
    only = df[idx][vars]
    scaler = StandardScaler()
    scaler.fit(only)
    return scaler

def fit_scaler_old(id_set, df, vars, sc):
    df = copy(df)
    idx = False
    for id in id_set:
        idx += df['instance_id'] == id
    only = df[idx][vars]
    df.loc[idx, vars] = sc.fit_transform(only)
    return df


def fit_scaler(id_set, df, vars):
    df_new = copy(df)
    for id in id_set:
        only = df.loc[df['instance_id'] == id, vars]
        scaler = MinMaxScaler()
        scaler.fit(only)
        df_new.loc[df['instance_id'] == id, vars] = scaler.fit_transform(only)
    return df_new

def make_stats(df, idx_list, input_vars):
    stats = np.array([])
    for id in idx_list:
        stats_tmp = df.loc[df['instance_id']==id, input_vars].describe().iloc[1:,:].to_numpy().flatten()
        stats = np.concatenate((stats, stats_tmp), axis=0)
    stats = pd.DataFrame(stats.reshape(-1,len(input_vars) * 7))
    stats['instance_id'] = idx_list
    return stats
def add_noise(original, std = 0.01):
    return original + std * torch.rand_like(original)

def random_masking(batch, mask_ratio=0.8, batch_max_mask_ratio=0.25):
    random_num = np.random.rand()
    if batch.ndim == 2:
        seq_len, input_dim = batch.shape
        if random_num > mask_ratio:
            batch[:, np.random.randint(input_dim)] = 0
    elif batch.ndim == 3:
        batch_size, seq_len, input_dim = batch.shape
        if random_num > mask_ratio:
            masked_batch_size = max(int(random_num * batch_max_mask_ratio),1)
            batch_masking = np.random.choice(range(batch_size), masked_batch_size)
            batch[batch_masking, :, np.random.randint(input_dim)] = 0
    return batch

def use_synthetic(df, input_vars, sim_idx,drawn_idx, sim_use=True, drawn_use=True):
    sc_sim = None
    sc_drawn = None

    if len(sim_idx) !=0:
        print('Valid simulated dataset')
        if sim_use:
            sc_sim = make_scaler(sim_idx, df, input_vars)
    if len(drawn_idx) !=0:
        print('Valid drawn dataset')
        if drawn_use:
            sc_drawn = make_scaler(drawn_idx, df, input_vars)
    return sc_sim, sc_drawn

def onehot_to_label(real, prediction):
    _, real_labels = torch.max(real, dim=1)
    _, pred_labels = torch.max(prediction, dim=1)
    real_labels = real_labels.detach().cpu().numpy()
    pred_labels = pred_labels.detach().cpu().numpy()
    colums_class = ['True', 'Pred']
    df_class = pd.DataFrame(columns=colums_class)
    df_class['True'] = real_labels
    df_class['Pred'] = pred_labels
    return df_class


'''
Visualization ##########################################################################################################
draw_loss: 학습결과가 저장된 logger를 입력받아 loss 혹은 accuracy를 시각화
draw_confusion: 예측 label과 실제 label 간의 confusion matrix를 시각화
draw_graph: 인스턴스 단위로 학습된 오터인코더의 경우, 실제 시계열값과 예측 시계열 값을 함께 시각화
########################################################################################################################
'''
def draw_loss(logger, view='loss', fname=None):
    plt.rcParams['figure.figsize'] = (8, 4)
    plt.rcParams['font.family'] = 'Times New Roman'
    if view == 'acc':
        train = logger['t_acc']
        val = logger['v_acc']
        loss = 'Accuracy'
    else:
        train = logger['t_loss']
        val = logger['v_loss']
        loss = 'Loss'
    e = range(1, len(logger['v_loss']) + 1)
    plt.plot(e, train, label='Train')
    plt.plot(e, val, label='Validation')
    plt.xlim([1, e[-1]])
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.legend()
    if fname:
        if not os.path.exists('./fig'):
            os.mkdir('./fig')
        plt.savefig(f'./fig/{fname}.png')
    plt.show()

def draw_confusion(df_class, events_names, fname=None):
    labels = [events_names[n] for n in list(set(df_class['True']))]
    plt.rcParams['figure.figsize'] = (6, 5)
    conf_m = confusion_matrix(df_class['True'], df_class['Pred'])
    sns.heatmap(
        conf_m, xticklabels=labels, yticklabels=labels,
        linewidths=.5, annot=True, cmap="YlGnBu", fmt="d")
    if fname:
        if not os.path.exists('./fig'):
            os.mkdir('./fig')
        plt.savefig(f'./fig/{fname}.png', bbox_inches='tight')
    plt.show()

def draw_graph(instance, prediction, real, input_vars, fname=None):
    plt.rcParams['figure.figsize'] = (8, 8)
    fig, ax = plt.subplots(len(input_vars), 1)
    for idx, var in enumerate(input_vars):
        ax[idx].plot(prediction[instance][:, idx], label='Prediction')
        ax[idx].plot(real[instance][:, idx], label='True')
        ax[idx].set_title(var)
        ax[idx].legend()
    plt.tight_layout()
    if fname:
        if not os.path.exists('./fig'):
            os.mkdir('./fig')
        plt.savefig(f'./fig/{fname}.png')
    plt.show()


'''
Model ##################################################################################################################
Encoder: LSTM 구조의 인코더   output => (hidden, cell) | [n_layers, batch_size, latend_dim]
Decoder: LSTM 구조의 디코더   output => reconstructed output (hidden, cell) | [batch_size, input_size]
Classifier: LSTM 인코더의 모든 층의 hidden state를 입력받아 label 분류 output => labels | [batch_size, n_features]
LSTMClassifier: 인코더, 디코터, 분류기를 포함하며, 사전 학습이 가능하도록 함
                - feature_extraction: Autoencoder 학습을 위한 것으로, teacher forcing, reversing이 적용됨
                - forward: Classification task를 수행함
########################################################################################################################
'''
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return (hidden, cell)

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)

        self.relu = nn.ReLU()
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/4)),
            #nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(hidden_size/4), output_size)
        )

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        prediction = self.fc(output)

        return prediction, (hidden, cell)

class Classifier(nn.Module):

    def __init__(self, hidden_size, n_features, num_layers=2):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.num_layers = num_layers
        self.input_size = hidden_size * num_layers

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, int(self.input_size/4)),
            nn.Dropout(0.2),
            nn.Linear(int(self.input_size/4), n_features)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        return self.fc(x)


class LSTMClassifier(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 n_features: int,
                 window_size: int = 1,
                 device = torch.device('cpu'),
                 **kwargs) -> None:
        """
        :param input_dim: 변수 Tag 갯수
        :param latent_dim: 최종 압축할 차원 크기
        :param window_size: 길이
        :param kwargs:
        """

        super(LSTMClassifier, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.window_size = window_size
        self.device = device
        if "num_layers" in kwargs:
            num_layers = kwargs.pop("num_layers")
        else:
            num_layers = 1

        self.encoder = Encoder(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_dim,
            output_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )
        self.classifier = Classifier(
            hidden_size=latent_dim,
            n_features=n_features,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        encoder_hidden = self.encoder(x)
        latent = encoder_hidden[0].transpose(0,1)
        # TODO Concat TSFRESH
        output = self.classifier(latent)
        return output

    def feature_extraction(self, x: torch.Tensor, **kwargs):
        batch_size, sequence_length, var_length = x.size()
        ## Encoder 넣기
        encoder_hidden = self.encoder(x)

        inv_idx = torch.arange(sequence_length - 1, -1, -1).long().to(self.device)
        inv_x = x.index_select(1, inv_idx).to(self.device)
        x_tmp = torch.cat((inv_x[:,1,:].reshape(batch_size,1,var_length), inv_x[:,1:,:]),dim=1)

        decoder_output, _ = self.reconstruct_decoder(x_tmp, encoder_hidden)
        reconstruct_output = decoder_output[:, inv_idx, :]
        return [reconstruct_output, x]

'''
Dataset ################################################################################################################
ByInstanceDataset: 인스턴스 단위로 데이터를 로딩시키기 위한 것으로, 인스턴스별 시퀀스 길이가 상이할 때 배치학습이 불가능
SlidingWindowDataset: Sliding window로 데이터를 로딩시키기 위한 것으로, 일정한 window size를 로딩
                    - window_size: x를 몇 개씩 사용할지
                    - stride: window를 몇 개씩 이동할지
                    - nan_treat: label에서 nan을 어떻게 처리할지 {ignore: 제거, back: backfill, forward: forwardfill}
    ※ is_binary: label이 multi class면 False
    ※ is_pretrain 키워드를 통하여 label을 사용할 지 결정할 수 있음 (Autoencoder 학습 시 True)
                    
########################################################################################################################
'''
class ByInstanceDataset(Dataset):
    def __init__(self, data, instance_id, all_instances, input_vars, stats, is_pretrain=False, is_binary=False, scaler=None, already_scaled=False):
        # dataframe: timestamp, 측정 변수들, class (Normal or Abnormal events), instance_id, source
        data = copy(data)
        # 2023-08-11: DGKim
        all_instances = copy(all_instances)

        self.scaler = scaler
        if not already_scaled:
            if scaler:
                self.data.loc[:, input_vars] = scaler.fit_transform(self.data[input_vars])
            else:
                self.data = fit_scaler(instance_id, data, input_vars)
        else:
            self.data = data
        self.instance_id = instance_id
        self.all_instances = all_instances
        self.input_vars = input_vars
        self.is_pretrain = is_pretrain

        if is_binary:
            all_instances.loc[all_instances['class_code'] != 0, 'class_code'] = 1
        self.class_one_hot = pd.get_dummies(all_instances['class_code'])

        self.stats = stats

    def __len__(self):
        return len(self.instance_id)

    def __getitem__(self, idx):
        self.x = self.data.loc[self.data['instance_id']==self.instance_id[idx], self.input_vars].to_numpy()

        if self.is_pretrain:
            return torch.FloatTensor(self.x)
        else:
            self.y = self.class_one_hot.loc[self.instance_id[idx]].to_numpy().squeeze()
            if isinstance(self.stats, pd.DataFrame):
                return torch.FloatTensor(self.x), torch.FloatTensor(self.stats.iloc[idx, :-1].to_numpy()), torch.FloatTensor(self.y)
            else:
                return torch.FloatTensor(self.x), torch.FloatTensor(self.y)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, instance_id, window_size, input_vars, stride=1, is_pretrain=False, is_binary=True, nan_treat='back', specific=False, scaler=None, already_scaled=False):
        # dataframe: timestamp, 측정 변수들, class (Normal, Transient, Abnormal), instance_id, source
        data = copy(data)
        # TODO Nan 값 처리 방법에 대해서 수정
        if is_binary:
            if nan_treat=='ignore':
                nan = data.isnull()['class']
                self.data = data[~nan]
                self.data.loc[self.data['class'] != 0, 'class'] = 1 # for binary classification
            elif nan_treat=='back':
                data.loc[:, 'class'].fillna(method='bfill')
                data.loc[data['class'] != 0, 'class'] = 1 # for binary classification
                self.data = data
            elif nan_treat=='forward':
                data.loc[:, 'class'].fillna(method='ffill')
                data.loc[data['class'] != 0, 'class'] = 1 # for binary classification
                self.data = data
            else:
                self.data = data
        else:
            if nan_treat=='ignore':
                nan = data.isnull()['class']
                self.data = data[~nan]
                self.data.loc[self.data['class'] >= 100, 'class'] =  self.data.loc[self.data['class'] >= 100, 'class'] - 100# for binary classification
            elif nan_treat=='back':
                data.loc[:, 'class'].fillna(method='bfill')
                data.loc[data['class'] >= 100, 'class'] = data.loc[data['class'] >= 100, 'class'] - 100# for binary classification

                self.data = data
            elif nan_treat=='forward':
                data.loc[:, 'class'].fillna(method='ffill')
                data.loc[data['class'] >= 0, 'class'] = data.loc[data['class'] >= 100, 'class'] - 100# for binary classification
                self.data = data
            else:
                self.data = data

        if not already_scaled:
            if scaler:
                self.data.loc[:, input_vars] = scaler.fit_transform(self.data[input_vars])
            else:
                self.data = fit_scaler(instance_id, data, input_vars)

        self.stride = stride

        # anomaly가 최초로 발생한 지점만 1, 나머지는 0
        if specific:
            start_idx = self.data[self.data['class'] ==1].index[0]
            specific_boolean = (self.data.index == start_idx)
            self.data.loc[specific_boolean, 'class'] = 1
            self.data.loc[~specific_boolean, 'class'] = 0

        self.instance_id = instance_id
        self.is_pretrain = is_pretrain
        self.window_size = window_size
        self.input_vars = input_vars
        self.jump = 0
        total_idx = 0
        self.total_len = 0
        for id in instance_id:
            crit = data['instance_id'] == id
            total_idx += (data['instance_id'] == id)
            if floor((sum(crit) - window_size - 1) / stride) >0:
                self.total_len += floor((sum(crit) - window_size - 1) / stride) + 1
        self.total_idx = total_idx.astype(bool)
        self.total_data = copy(self.data[self.total_idx])

        self.one_hot_class = pd.get_dummies(self.data['class'])[self.total_idx]

    def __len__(self):
        return int(self.total_len)


    def __getitem__(self, idx_tmp):
        if idx_tmp == 0:
            self.jump = 0

        idx = self.stride * idx_tmp + self.jump
        if self.total_data.iloc[idx, ::]['instance_id'] != self.total_data.iloc[idx + self.window_size, ::]['instance_id']:
            gap = 1
            while True:
                if self.total_data.iloc[idx + gap, ::]['instance_id'] == self.total_data.iloc[idx + self.window_size, ::]['instance_id']:
                    self.jump += gap
                    break
                else:
                    gap += 1
            idx = self.stride * idx_tmp + self.jump

        self.x = self.total_data[self.input_vars].iloc[idx:idx + self.window_size, ::].to_numpy()
        if self.is_pretrain:
            return torch.FloatTensor(self.x)
        else:
            self.y = self.one_hot_class.iloc[idx + self.window_size, ::].to_numpy().squeeze()
            return torch.FloatTensor(self.x), torch.FloatTensor(self.y)

# class SlidingWindowDataset(Dataset):
#     def __init__(self, data, instance_id, window_size, input_vars, is_pretrain=False, nan_treat='back', specific=False, scaler=None):
#         # dataframe: timestamp, 측정 변수들, class (Normal, Transient, Abnormal), instance_id, source
#         # TODO Nan 값 처리 방법에 대해서 수정
#         if nan_treat=='ignore':
#             nan = data.isnull()['class']
#             self.data = data[~nan]
#             self.data.loc[self.data['class'] != 0, 'class'] = 1 # for binary classification
#         elif nan_treat=='back':
#             data.loc[:, 'class'].fillna(method='bfill')
#             data.loc[data['class'] != 0, 'class'] = 1 # for binary classification
#             self.data = data
#         elif nan_treat=='forward':
#             data.loc[:, 'class'].fillna(method='ffill')
#             data.loc[data['class'] != 0, 'class'] = 1 # for binary classification
#             self.data = data
#         else:
#             self.data = data
#         if scaler:
#             self.data.loc[:, input_vars] = scaler.fit_transform(self.data[input_vars])
#
#         # anomaly가 최초로 발생한 지점만 1, 나머지는 0
#         if specific:
#             start_idx = self.data[self.data['class'] ==1].index[0]
#             specific_boolean = (self.data.index == start_idx)
#             self.data.loc[specific_boolean, 'class'] = 1
#             self.data.loc[~specific_boolean, 'class'] = 0
#
#         self.is_pretrain = is_pretrain
#         self.window_size = window_size
#         self.input_vars = input_vars
#         self.jump = 0
#         total_idx = 0
#
#         for id in instance_id:
#             total_idx += (data['instance_id'] == id)
#         self.total_idx = total_idx.astype(bool)
#         self.total_data = copy(self.data[self.total_idx])
#         self.total_len = int(sum(total_idx) - window_size * len(instance_id))
#
#         self.one_hot_class = pd.get_dummies(self.data['class'])[self.total_idx]
#
#     def __len__(self):
#         return self.total_len
#
#     def __getitem__(self, idx_tmp):
#         if idx_tmp == 0:
#             self.jump = 0
#         idx = idx_tmp + self.jump
#         if self.total_data.iloc[idx, ::]['instance_id'] != self.total_data.iloc[idx + self.window_size, ::]['instance_id']:
#             self.jump += self.window_size
#             idx = idx_tmp + self.jump
#
#         self.x = self.total_data[self.input_vars].iloc[idx:idx + self.window_size, ::].to_numpy()
#         if self.is_pretrain:
#             return torch.FloatTensor(self.x)
#         else:
#             self.y = self.one_hot_class.iloc[idx + self.window_size, ::].to_numpy().squeeze()
#             return torch.FloatTensor(self.x), torch.FloatTensor(self.y)
'''
Setting ################################################################################################################
Setting_for_Pretrain: autoencoder를 이용한 self-supervised learning에 필요한 사전준비
                    - is_full: 인스턴스별 full sequence를 넣어줄지 (True면, batch size = 1)
    
Setting_for_Classifier: Classifier 모델을 학습하기 위한 사전준비
                    - class_code: abnormal class를 정수 혹은 리스트로 전달받으면, 해당 문제로만 데이터로더 구성
                    - is_binary: label을 바이너리로 할지 결정
                    - is_full: 인스턴스별 full sequence를 넣어줄지 (True면, batch size = 1)

    <<output>>
    - scaler_dict: source에 따른 스케일러 반환 {Real, Simulate, Drawn}
    - dataloader_dict: 용도에 따른 데이터로더 반환 {Train, Valid, Test}
########################################################################################################################
'''
def Setting_for_Pretrain(args, df, df_original, input_vars, instances, real_instances, add_instances=None,
                         scaler_global=False, is_full=True, already_scaled=False, use_stats=False):

    train_id, valid_id, test_id = [], [], []
    real_idx_dict = {}
    for class_code in set(real_instances['class_code']):
        real_idx_dict[class_code] = list(real_instances[real_instances['class_code'] == class_code].index.values)

    for class_code in set(real_instances['class_code']):
        # 2023-08-11: Compatible with Simulated and Drawn dataset
        if ceil(args.test_ratio * len(real_instances[real_instances['class_code'] == class_code])) <= 1:
            class_tmp = [int(real_idx_dict[class_code].pop(np.random.randint(len(real_idx_dict[class_code]))))]
            test_id += class_tmp
            train_valid_id = real_idx_dict[class_code]
            if isinstance(add_instances, pd.DataFrame):
                train_valid_id += list(add_instances[add_instances['class_code'] == class_code].index.values)
            train_id_class, valid_id_class = train_test_split(train_valid_id, test_size=args.validation_ratio/(1-args.test_ratio))
            train_id += train_id_class
            valid_id += valid_id_class
        else:
            test_id_class = np.random.choice(real_idx_dict[class_code], size=ceil(args.test_ratio * len(real_instances[real_instances['class_code'] == class_code])), replace=False)
            test_id += list(test_id_class)
            train_valid_id = list(set(real_idx_dict[class_code]) - set(test_id_class))
            if isinstance(add_instances, pd.DataFrame):
               train_valid_id += list(add_instances[add_instances['class_code'] == class_code].index.values)
            train_id_class, valid_id_class = train_test_split(train_valid_id, test_size=args.validation_ratio/(1-args.test_ratio))
            train_id += train_id_class
            valid_id += valid_id_class
    train_id = sorted(train_id)
    valid_id = sorted(valid_id)
    test_id = sorted(test_id)

    scaler_dict = {}
    stats = {}
    if use_stats:
        stats['Train'] = make_stats(df_original, train_id, input_vars)
        stats['Valid'] = make_stats(df_original, valid_id, input_vars)
        stats['Test'] = make_stats(df_original, test_id, input_vars)
        stats_col = stats['Train'].columns[:-1]
        scaler_dict['Stats'] = make_stats_scaler(train_id, stats['Train'], stats_col)
        stats['Train'] = fit_scaler_old(train_id, stats['Train'], stats_col, scaler_dict['Stats'])
        stats['Valid'] = fit_scaler_old(valid_id, stats['Valid'], stats_col, scaler_dict['Stats'])
        stats['Test'] = fit_scaler_old(test_id, stats['Test'], stats_col, scaler_dict['Stats'])
    else:
        stats['Train'], stats['Valid'], stats['Test'] = None, None, None

    if scaler_global:
        scaler_dict['Real'] = make_scaler(train_id, df, input_vars)
        scaler_dict['Simulate'], scaler_dict['Drawn'] = None, None
    #     scaler_dict['Simulate'], scaler_dict['Drawn'] = use_synthetic(df, input_vars, sim_idx, drawn_idx,
    #                                                                   sim_use=args.sim_use, drawn_use=args.drawn_use)
    else:
        scaler_dict['Real'], scaler_dict['Simulate'], scaler_dict['Drawn'] = None, None, None
    if is_full:
        train_dataset = ByInstanceDataset(data=df,instance_id=train_id, all_instances=instances, input_vars=input_vars,
                                          stats = stats['Train'], is_pretrain=args.is_pretrain,
                                          scaler=scaler_dict['Real'], already_scaled=already_scaled)
        valid_dataset = ByInstanceDataset(data=df, instance_id=valid_id, all_instances=instances, input_vars=input_vars,
                                          stats = stats['Valid'], is_pretrain=args.is_pretrain,
                                          scaler=scaler_dict['Real'], already_scaled=already_scaled)
        test_dataset = ByInstanceDataset(data=df, instance_id=test_id, all_instances=instances, input_vars=input_vars,
                                         stats = stats['Test'], is_pretrain=args.is_pretrain,
                                         scaler=scaler_dict['Real'], already_scaled=already_scaled)
        batch_size = 1
    else:
        train_dataset = SlidingWindowDataset(data=df, instance_id=train_id, window_size=args.window_size,stride=args.stride,
                                                    input_vars=input_vars, is_pretrain=True, scaler=scaler_dict['Real'], already_scaled=already_scaled)
        valid_dataset = SlidingWindowDataset(data=df, instance_id=valid_id, window_size=args.window_size,stride=args.stride,
                                                    input_vars=input_vars, is_pretrain=True, scaler=scaler_dict['Real'], already_scaled=already_scaled)
        test_dataset = SlidingWindowDataset(data=df, instance_id=test_id, window_size=args.window_size,stride=args.stride,
                                                   input_vars=input_vars, is_pretrain=True, scaler=scaler_dict['Real'], already_scaled=already_scaled)
        batch_size = args.batch_size
    id_dict = {}
    id_dict['Train'] = train_id
    id_dict['Valid'] = valid_id
    id_dict['Test'] = test_id

    dataloader_dict = {}
    dataloader_dict['Train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dataloader_dict['Valid'] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    dataloader_dict['Test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Train: {len(train_dataset)} \nValid: {len(valid_dataset)}\nTest: {len(test_dataset)}')
    return scaler_dict, dataloader_dict, id_dict

def Setting_for_Classifier(args, df, df_original, input_vars, instances, real_instances, add_instances=None,
                           scaler_global=False, is_binary=False, is_full=False, class_list=None, already_scaled=True,
                           use_stats=True):
    if class_list:
        real_instances = copy(real_instances)
        instances = copy(instances)
        real = 0
        for code in class_list:
            real += (real_instances['class_code'] == code)
        real_instances = real_instances[real.astype(bool).to_numpy()]

    train_id, valid_id, test_id = [], [], []
    real_idx_dict = {}
    for class_code in set(real_instances['class_code']):
        real_idx_dict[class_code] = list(real_instances[real_instances['class_code'] == class_code].index.values)

    for class_code in set(real_instances['class_code']):
        # 2023-08-11: Only for Real instances
        if ceil(args.test_ratio * len(real_instances[real_instances['class_code'] == class_code])) <= 1:
            class_tmp = [int(real_idx_dict[class_code].pop(np.random.randint(len(real_idx_dict[class_code]))))]
            test_id += class_tmp
            train_valid_id = real_idx_dict[class_code]
            if isinstance(add_instances, pd.DataFrame):
                train_valid_id += list(add_instances[add_instances['class_code'] == class_code].index.values)
            train_id_class, valid_id_class = train_test_split(train_valid_id,
                                                              test_size=args.validation_ratio / (1 - args.test_ratio))
            train_id += train_id_class
            valid_id += valid_id_class
        else:
            test_id_class = np.random.choice(real_idx_dict[class_code], size=ceil(
                args.test_ratio * len(real_instances[real_instances['class_code'] == class_code])), replace=False)

            test_id += list(test_id_class)
            train_valid_id = list(set(real_idx_dict[class_code]) - set(test_id_class))
            if isinstance(add_instances, pd.DataFrame):
                train_valid_id += list(add_instances[add_instances['class_code'] == class_code].index.values)
            train_id_class, valid_id_class = train_test_split(train_valid_id,
                                                              test_size=args.validation_ratio / (1 - args.test_ratio))
            train_id += train_id_class
            valid_id += valid_id_class

        train_id = sorted(train_id)
        valid_id = sorted(valid_id)
        test_id = sorted(test_id)

    scaler_dict = {}
    stats = {}
    if use_stats:
        stats['Train'] = make_stats(df_original, train_id, input_vars)
        stats['Valid'] = make_stats(df_original, valid_id, input_vars)
        stats['Test'] = make_stats(df_original, test_id, input_vars)
        stats_col = stats['Train'].columns[:-1]
        scaler_dict['Stats'] = make_stats_scaler(train_id, stats['Train'], stats_col)
        stats['Train'] = fit_scaler_old(train_id, stats['Train'], stats_col, scaler_dict['Stats'])
        stats['Valid'] = fit_scaler_old(valid_id, stats['Valid'], stats_col, scaler_dict['Stats'])
        stats['Test'] = fit_scaler_old(test_id, stats['Test'], stats_col, scaler_dict['Stats'])
    else:
        stats['Train'], stats['Valid'], stats['Test'] = None, None, None

    if scaler_global:
        scaler_dict['Real'] = make_scaler(train_id, df, input_vars)
        scaler_dict['Simulate'], scaler_dict['Drawn'] = None, None
        # scaler_dict['Simulate'], scaler_dict['Drawn'] = use_synthetic(df, input_vars, sim_idx, drawn_idx,
        #                                                               sim_use=args.sim_use, drawn_use=args.drawn_use)
    else:
        scaler_dict['Real'], scaler_dict['Simulate'], scaler_dict['Drawn'] = None, None, None

    if not is_full:
        train_dataset = SlidingWindowDataset(data=df, instance_id=train_id, window_size=args.window_size, stride=args.stride,
                                             input_vars=input_vars, is_binary= is_binary, scaler=scaler_dict['Real'], already_scaled=already_scaled)
        valid_dataset = SlidingWindowDataset(data=df, instance_id=valid_id, window_size=args.window_size, stride=args.stride,
                                             input_vars=input_vars, is_binary= is_binary, scaler=scaler_dict['Real'], already_scaled=already_scaled)
        test_dataset = SlidingWindowDataset(data=df, instance_id=test_id, is_binary= is_binary, stride=args.stride,
                                            window_size=args.window_size, input_vars=input_vars,scaler=scaler_dict['Real'], already_scaled=already_scaled)

        batch_size = args.batch_size
    else:
        train_dataset = ByInstanceDataset(data=df, instance_id=train_id, all_instances=instances, input_vars=input_vars,
                                          stats=stats['Train'], scaler=scaler_dict['Real'], is_binary= is_binary,
                                          already_scaled=already_scaled)
        valid_dataset = ByInstanceDataset(data=df, instance_id=valid_id, all_instances=instances, input_vars=input_vars,
                                          stats=stats['Valid'], scaler=scaler_dict['Real'], is_binary= is_binary,
                                          already_scaled=already_scaled)
        test_dataset = ByInstanceDataset(data=df, instance_id=test_id, all_instances=instances, input_vars=input_vars,
                                         stats=stats['Test'], scaler=scaler_dict['Real'], is_binary= is_binary,
                                         already_scaled=already_scaled)
        batch_size = 1

    id_dict = {}
    id_dict['Train'] = train_id
    id_dict['Valid'] = valid_id
    id_dict['Test'] = test_id
    dataloader_dict = {}
    dataloader_dict['Train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dataloader_dict['Valid'] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    dataloader_dict['Test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Train: {len(train_dataset)} \nValid: {len(valid_dataset)}\nTest: {len(test_dataset)}')
    return scaler_dict, dataloader_dict, id_dict

'''
Training ###############################################################################################################
Pretraining: self-supervised learning with LSTM AE             output => trained model, logger[loss]
Train_Classifier: supervised learning with LSTM classifier     output => trained model, logger[loss, accuarcy]
    ※ Vicinal risk minimization: 데이터에 random noise를 추가하여 augmentation
    ※ Masknig: simulated data와 drawn data, 그리고 real data 중 frozen이 없는 데이터를 모사하기 위함
########################################################################################################################
'''
def Pretraining(param, model, criterion, optimizer, scheduler, dataloader_dict, model_name='mdl'):
    logger = {'t_loss':[], 'v_loss':[]}
    min_valid_loss = 1e8
    iterator = tqdm(range(param.num_epoch))
    model.to(param.device)
    for epoch in iterator:
        model.train()
        len_tmp = 0
        train_loss_tmp = 0.0
        for batch in dataloader_dict['Train']:
            x = batch
            if param.masking:
                x = random_masking(x,mask_ratio=param.mask_ratio,
                                   batch_max_mask_ratio=param.batch_max_mask_ratio)
            if param.vicinal_risk_minimization:
                noised_x = add_noise(x)
            else:
                noised_x = x

            noised_x = noised_x.to(param.device)
            pred = model.feature_extraction(noised_x)
            loss = criterion(*pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            len_tmp += len(x) # Accuracy calculation
            train_loss_tmp += loss.item() # Loss calculation
        train_loss_tmp /= len_tmp # Loss calculation
        logger['t_loss'].append(train_loss_tmp)

        model.eval()
        with torch.no_grad():
            valid_loss_tmp= 0.0
            len_tmp = 0
            for batch in dataloader_dict['Valid']:
                x= batch
                x = x.to(param.device)
                pred = model.feature_extraction(x)
                loss = criterion(*pred)
                len_tmp += len(x) # Accuracy calculation
                valid_loss_tmp += loss.item()
            valid_loss_tmp /= len_tmp # Loss calculation
            logger['v_loss'].append(valid_loss_tmp)

            scheduler.step()
            if min_valid_loss > logger['v_loss'][-1]:
                min_valid_loss = logger['v_loss'][-1]
                if not os.path.exists('./cache'):
                    os.mkdir('./cache')
                torch.save(model.state_dict(), f'./cache/{model_name}.pth')

        if (epoch % 1 == 0) or (epoch == param.num_epoch - 1):
            print("Epoch: %d, train loss: %1.5f, valid loss: %1.5f" % (epoch+1,logger['t_loss'][-1], logger['v_loss'][-1],))
    model.load_state_dict(torch.load(f'./cache/{model_name}.pth'))
    return model, logger


def Train_Classifier(param, model, criterion, optimizer, scheduler, dataloader_dict, model_name='mdl'):
    logger = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
    min_valid_loss = 1e8
    model.to(param.device)
    iterator = tqdm(range(param.num_epoch))
    for epoch in iterator:
        model.train()
        train_acc_tmp = 0
        len_tmp = 0
        train_loss_tmp = 0.0
        for batch in dataloader_dict['Train']:
            x, y = batch
            y = y.to(param.device)
            if param.vicinal_risk_minimization:
                # x = add_noise(x, std = 0.2)
                x = add_noise(x, std=0.01)
            if param.masking:
                x = random_masking(x,
                                   mask_ratio=param.mask_ratio,
                                   batch_max_mask_ratio=param.batch_max_mask_ratio)
            x = x.to(param.device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred_labels = torch.max(pred, dim=1)
            _, true_labels = torch.max(y, dim=1)
            train_acc_tmp += torch.sum(pred_labels == true_labels)  # Accuracy calculation
            len_tmp += len(x)  # Accuracy calculation
            train_loss_tmp += loss.item()  # Loss calculation
        train_loss_tmp /= len_tmp  # Loss calculation
        logger['t_loss'].append(train_loss_tmp)
        logger['t_acc'].append(float(train_acc_tmp / len_tmp))  # Accuracy calculation

        model.eval()
        with torch.no_grad():
            valid_loss_tmp = 0.0
            valid_acc_tmp = 0
            len_tmp = 0
            for batch in dataloader_dict['Valid']:
                x, y = batch
                x = x.to(param.device)
                y = y.to(param.device)
                pred = model(x)
                loss = criterion(pred, y)
                _, pred_labels = torch.max(pred, dim=1)
                _, true_labels = torch.max(y, dim=1)
                valid_acc_tmp += torch.sum(pred_labels == true_labels)  # Accuracy calculation
                len_tmp += len(x)  # Accuracy calculation
                valid_loss_tmp += loss.item()  # Loss calculation
            valid_loss_tmp /= len_tmp  # Loss calculation
            logger['v_loss'].append(valid_loss_tmp)
            logger['v_acc'].append(float(valid_acc_tmp / len_tmp))  # Accuracy calculation

        scheduler.step()
        if min_valid_loss > logger['v_loss'][-1]:
            min_valid_loss = logger['v_loss'][-1]
            if not os.path.exists('./cache'):
                os.mkdir('./cache')
            torch.save(model.state_dict(), f'./cache/{model_name}.pth')

        if (epoch % 5 == 0) or (epoch == param.num_epoch - 1):
            print("Epoch: %d, train loss: %1.5f, valid loss: %1.5f, train acc: %1.3f, valid acc: %1.3f" % (epoch+1,
                                                                                                           logger['t_loss'][-1],
                                                                                                           logger['v_loss'][-1],
                                                                                                           logger['t_acc'][-1],
                                                                                                           logger['v_acc'][-1]))
    model.load_state_dict(torch.load(f'./cache/{model_name}.pth'))
    return model, logger


'''
Inference ##############################################################################################################
Inference_AE: LSTM AE를 test dataloader로 검증
Inference_Classifier: LSTM classifier를 test dataloader로 검증
########################################################################################################################
'''
def Inference_AE(model, dataloader):
    predictions = []
    reals = []
    predict = {}
    real = {}
    model.eval()
    len_tmp = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x= batch
            x = x.to(model.device)
            pred = model.feature_extraction(x)
            predictions.extend(pred[0].detach().cpu().numpy().flatten())
            reals.extend(x.detach().cpu().numpy().flatten())
            predict[idx] = pred[0].detach().cpu().numpy().squeeze()
            real[idx] = x.detach().cpu().numpy().squeeze()
    print(f'R2: {r2_score(reals,predictions):.3f}')
    return predict, real

def Inference_Classifier(model, dataloader, criterion):
    model.to(model.device)
    model.eval()
    test_loss = []
    test_acc = []
    test_loss_tmp= 0.0
    test_acc_tmp = 0
    test_f1_tmp = 0
    len_f1 = 0
    len_tmp = 0
    reals = torch.FloatTensor([]).to(model.device)
    predictions = torch.FloatTensor([]).to(model.device)
    for batch in dataloader:
        x, y = batch
        x = x.to(model.device)
        y = y.to(model.device)
        pred = model(x)
        loss = criterion(pred, y)
        _, pred_labels = torch.max(pred, dim=1)
        _, true_labels = torch.max(y, dim=1)
        test_acc_tmp += torch.sum(pred_labels == true_labels) # Accuracy calculation
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels.detach().cpu().numpy(),
                                                                   pred_labels.detach().cpu().numpy(),
                                                                   average='micro')
        test_f1_tmp += f1
        len_f1 += 1
        len_tmp += len(x) # Accuracy calculation
        test_loss_tmp += loss.item() # Loss calculation
        reals = torch.concat([reals, y])
        predictions = torch.concat([predictions, pred])
    test_loss_tmp /= len_tmp # Loss calculation
    test_loss.append(test_loss_tmp)
    test_acc.append(float(test_acc_tmp / len_tmp)) # Accuracy calculation
    f1 = test_f1_tmp / len_f1
    print(f'Test loss: {test_loss[-1]:.3f} |Test accuracy: {test_acc[-1]:.3f}  |F1-Score: {np.average(f1):.3f}')
    return predictions, reals

'''
PCA & Random Forest ####################################################################################################
Decompose: LSTM encoder를 통해 구해진 latent vector를 kernel PCA로 3차원으로 축소
RandomForest: 차원축소된 latent vector를 이용하여 random forest 학습
########################################################################################################################
'''
def Decompose(model, df, id_use, instances, input_vars, scaler, color, view=False):
    dataset = ByInstanceDataset(data=df, instance_id=id_use, all_instances=instances, is_binary=False,
                      input_vars=input_vars, scaler=scaler, is_pretrain=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    lst = {}
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = batch.to(model.device)
            hidden_tmp = model.encoder(batch)[0]
            lst[idx] = hidden_tmp.detach().cpu().numpy().flatten()
    d = pd.DataFrame(lst).T

    pca = KernelPCA(n_components=3, kernel='rbf')
    pca.fit(d)
    decomp_d = pca.fit_transform(d)
    if view:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(decomp_d[:, 0], decomp_d[:, 1], decomp_d[:, 2], c=color)
    return decomp_d, pca

def RandomForest(decomposed_data, class_code, hyper, parameter_opt=True, test_size=0.3, seed=0, verbose=2):
    def _objective(n_estimators, max_depth, min_samples_leaf, min_samples_split):
        hyper_type = round
        for param in hyper.keys():
            hyper[param] = hyper_type(eval(param))
        RF_model = RandomForestClassifier(n_estimators=int(n_estimators),
                                         max_depth=int(max_depth),
                                         min_samples_leaf=int(min_samples_leaf),
                                         min_samples_split=int(min_samples_split),
                                          random_state=seed)
        RF_model.fit(X_train, y_train)
        y_pred = RF_model.predict(X_test)
        colums_class = ['True', 'Pred']
        df_class = pd.DataFrame(columns=colums_class)
        df_class['True'] = y_test
        df_class['Pred'] = y_pred
        ret = precision_recall_fscore_support(df_class['True'], df_class['Pred'], average='micro')
        p, r, f1, _ = ret
        return np.average(f1)

    X_train, X_test, y_train, y_test = train_test_split(decomposed_data, class_code, test_size=test_size,random_state=seed)
    if parameter_opt:
        BO = BayesianOptimization(f=_objective, pbounds=hyper,random_state=seed, verbose=verbose)
        acquisition_function = UtilityFunction(kind='ei', xi=0.0001)
        BO.maximize(init_points=10, n_iter=40, acquisition_function=acquisition_function)
        params_opt = {}
        for param in BO.max['params'].keys():
            params_opt[param] = int(BO.max['params'][param])
    else:
        params_opt = {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 5}



    RF_model = RandomForestClassifier(n_estimators=params_opt['n_estimators'],
                                      max_depth=params_opt['max_depth'],
                                      min_samples_leaf=params_opt['min_samples_leaf'],
                                      min_samples_split=params_opt['min_samples_split'],
                                      random_state=seed)
    RF_model.fit(X_train, y_train)
    y_pred = RF_model.predict(X_test)
    colums_class = ['True', 'Pred']
    df_class = pd.DataFrame(columns=colums_class)
    df_class['True'] = y_test
    df_class['Pred'] = y_pred
    precision, recall, f1, _ = precision_recall_fscore_support(df_class['True'], df_class['Pred'], average='micro')

    # print('{:>10} : {:.2f}'.format( 'accuracy', accuracy ) )
    if verbose <= 3:
        print('{:>10} : {:.3f}'.format('precision', np.average(precision)))
        print('{:>10} : {:.3f}'.format('recall', np.average(recall)))
        print('{:>10} : {:.3f}'.format('f1', np.average(f1)))
    return RF_model, df_class