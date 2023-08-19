import pandas as pd
import numpy as np
import logging
import warnings
import sys
import pickle
sys.path.append('stac')
from time import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
logging.getLogger('tsfresh').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore')
import random
import torch
import os
from copy import copy
def class_and_file_generator(data_path, real=False, simulated=False, drawn=False):
    for class_path in data_path.iterdir():
        if class_path.is_dir():
            class_code = int(class_path.stem)
            for instance_path in class_path.iterdir():
                if (instance_path.suffix == '.csv'):
                    if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                       (drawn and instance_path.stem.startswith('DRAWN')) or \
                       (real and (not instance_path.stem.startswith('SIMULATED')) and \
                       (not instance_path.stem.startswith('DRAWN'))):
                        yield class_code, instance_path


def get_instances_with_undesirable_event(data_path, undesirable_event_code,
                                         real, simulated, drawn):
    instances = pd.DataFrame(class_and_file_generator(data_path,
                                                      real=real,
                                                      simulated=simulated,
                                                      drawn=drawn),
                             columns=['class_code', 'instance_path'])
    idx = instances['class_code'] == undesirable_event_code
    return instances.loc[idx].reset_index(drop=True)

def load_instance(instance_path, args):
    try:
        well, instance_id = instance_path.stem.split('_')
        df = pd.read_csv(instance_path, sep=',', header=0)
        assert (df.columns == args.columns).all(), 'invalid args.columns in the file {}: {}'\
            .format(str(instance_path), str(df.columns.tolist()))
        return df
    except Exception as e:
        raise Exception('error reading file {}: {}'.format(instance_path, e))

def load_and_downsample_instances(instances, downsample_rate, source, instance_id, args):
    df_instances = pd.DataFrame()
    for _, row in instances.iterrows():
        _, instance_path = row
        df = load_instance(instance_path, args).iloc[::downsample_rate, :]
        df['instance_id'] = instance_id
        instance_id += 1
        df_instances = pd.concat([df_instances, df])
    df_instances['source'] = source
    return df_instances.reset_index(drop=True), instance_id


def extract_samples_train(df, df_samples_train, df_y_train, sample_id, args):
    instance = df['instance_id'].iloc[0]
    f_idxs = []
    l_idxs = []

    # Gets the observations labels and their unequivocal set
    ols = list(df['class'])
    set_ols = set()
    for ol in ols:
        if ol in set_ols or np.isnan(ol):
            continue
        set_ols.add(int(ol))

    # Discards the source and the observations labels and replaces all nan with
    # 0 (tsfresh's requirement)
    df_vars = df.drop(['source', 'class'], axis=1).fillna(0)

    # Extracts samples from the normal period (if it exists)
    #
    if args.normal_class_code in set_ols:
        # Gets indexes (first and last) without overlap with other periods
        f_idx = ols.index(args.normal_class_code)
        l_idx = len(ols) - 1 - ols[::-1].index(args.normal_class_code)

        # Defines the proper step and extracts samples
        max_samples = l_idx - f_idx + 1 - args.sample_size_normal_period
        if (max_samples) > 0:
            num_samples = min(args.max_samples_per_period, max_samples)
            if num_samples == max_samples:
                step_max = 1
            else:
                step_max = (max_samples - 1) // (args.max_samples_per_period - 1)
            step_wanted = args.sample_size_normal_period
            step = min(step_wanted, step_max)

            # Extracts samples
            for idx in range(num_samples):
                f_idx_c = l_idx - args.sample_size_normal_period + 1 - (num_samples - 1 - idx) * step
                l_idx_c = f_idx_c + args.sample_size_normal_period
                f_idxs.append(f_idx_c)
                l_idxs.append(l_idx_c)
                # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
                df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
                df_sample.insert(loc=0, column='id', value=sample_id)
                df_samples_train = df_samples_train.append(df_sample)
                df_y_train = df_y_train.append({'instance': instance,
                                                'y': args.normal_class_code},
                                               ignore_index=True)
                sample_id += 1

    # Extracts samples from the transient period (if it exists)
    #
    transient_code = args.undesirable_event_code + 100
    if transient_code in set_ols:
        # Gets indexes (first and last) with possible overlap at the beginning
        # of this period
        f_idx = ols.index(transient_code)
        if f_idx - (args.sample_size_default - 1) > 0:
            f_idx = f_idx - (args.sample_size_default - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(transient_code)

        # Defines the proper step and extracts samples
        max_samples = l_idx - f_idx + 1 - args.sample_size_default
        if (max_samples) > 0:
            num_samples = min(args.max_samples_per_period, max_samples)
            if num_samples == max_samples:
                step_max = 1
            else:
                step_max = (max_samples - 1) // (args.max_samples_per_period - 1)
            step_wanted = np.inf
            step = min(step_wanted, step_max)

            # Extracts samples
            for idx in range(num_samples):
                f_idx_c = f_idx + idx * step
                l_idx_c = f_idx_c + args.sample_size_default
                f_idxs.append(f_idx_c)
                l_idxs.append(l_idx_c)
                # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
                df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
                df_sample.insert(loc=0, column='id', value=sample_id)
                df_samples_train = df_samples_train.append(df_sample)
                df_y_train = df_y_train.append({'instance': instance,
                                                'y': transient_code},
                                               ignore_index=True)
                sample_id += 1

    # Extracts samples from the in-regime period (if it exists)
    #
    if args.undesirable_event_code in set_ols:
        # Gets indexes (first and last) with possible overlap at the beginning
        # or end of this period
        f_idx = ols.index(args.undesirable_event_code)
        if f_idx - (args.sample_size_default - 1) > 0:
            f_idx = f_idx - (args.sample_size_default - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(args.undesirable_event_code)
        if l_idx + (args.sample_size_default - 1) < len(ols) - 1:
            l_idx = l_idx + (args.sample_size_default - 1)
        else:
            l_idx = len(ols) - 1

        # Defines the proper step and extracts samples
        max_samples = l_idx - f_idx + 1 - args.sample_size_default
        if (max_samples) > 0:
            num_samples = min(args.max_samples_per_period, max_samples)
            if num_samples == max_samples:
                step_max = 1
            else:
                step_max = (max_samples - 1) // (args.max_samples_per_period - 1)
            step_wanted = args.sample_size_default
            step = min(step_wanted, step_max)

            # Extracts samples
            for idx in range(num_samples):
                f_idx_c = f_idx + idx * step
                l_idx_c = f_idx_c + args.sample_size_default
                f_idxs.append(f_idx_c)
                l_idxs.append(l_idx_c)
                # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
                df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
                df_sample.insert(loc=0, column='id', value=sample_id)
                df_samples_train = df_samples_train.append(df_sample)
                df_y_train = df_y_train.append({'instance': instance,
                                                'y': args.undesirable_event_code},
                                               ignore_index=True)
                sample_id += 1

    return df_samples_train, df_y_train, sample_id


def extract_samples_test(df, df_samples_test, df_y_test, sample_id, args):
    instance = df['instance_id'].iloc[0]
    f_idxs = []
    l_idxs = []

    # Gets the observations labels
    ols = list(df['class'].fillna(method='ffill'))

    # Discards the source and the observations labels and replaces all nan with
    # 0 (tsfresh's requirement)
    df_vars = df.drop(['source', 'class'], axis=1).fillna(0)

    # Extracts samples from the instance as a whole
    f_idx = 0
    l_idx = len(df) - 1

    # Defines the proper step and extracts samples
    max_samples = l_idx - f_idx + 1 - args.sample_size_default
    if (max_samples) > 0:
        num_samples = min(3 * args.max_samples_per_period, max_samples)
        if num_samples == max_samples:
            step_max = 1
        else:
            step_max = (max_samples - 1) // (3 * args.max_samples_per_period - 1)
        step_wanted = np.inf
        step = min(step_wanted, step_max)

        # Extracts samples
        for idx in range(num_samples):
            f_idx_c = f_idx + idx * step
            l_idx_c = f_idx_c + args.sample_size_default
            f_idxs.append(f_idx_c)
            l_idxs.append(l_idx_c)
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            df_y_test = df_y_test.append({'instance': instance, 'y': ols[l_idx_c]},
                                         ignore_index=True)
            sample_id += 1

    return df_samples_test, df_y_test, sample_id


def train_test_calc_scores(X_train, y_train, X_test, y_test, scores, clfs, scenario):
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    for clf_name, clf in clfs.items():
        try:
            # Train
            t0 = time()
            clf.fit(X_train, y_train)
            t_train = time() - t0

            # Test
            t0 = time()
            y_pred = clf.predict(X_test)
            t_test = time() - t0

            # Plots actual and predicted labels
            # fig = plt.figure(figsize=(12,1))
            # ax = fig.add_subplot(111)
            # plt.plot(y_pred, marker=11, color='orange', linestyle='')
            # plt.plot(y_test, marker=10, color='green', linestyle='')
            # ax.grid(False)
            # ax.set_yticks([0, args.undesirable_event_code])
            # ax.set_yticklabels([args.normal_class_code, args.undesirable_event_code])
            # ax.set_title(clf_name)
            # ax.set_xlabel('Sample')
            # ax.legend(['Predicted labels', 'Actual labels'])
            # plt.show()

            # Calculates the considered scores
            ret = precision_recall_fscore_support(y_test, y_pred, average='micro')
            p, r, f1, _ = ret
            scores = scores.append({'SCENARIO': scenario,
                                    'CLASSIFIER': clf_name,
                                    'PRECISION': p,
                                    'RECALL': r,
                                    'F1': f1,
                                    'TRAINING[s]': t_train,
                                    'TESTING[s]': t_test}, ignore_index=True)

        except Exception as e:
            print('error in training/testing classifier: {}'.format(e))
            scores = scores.append({'SCENARIO': scenario,
                                    'CLASSIFIER': clf_name,
                                    'PRECISION': np.nan,
                                    'RECALL': np.nan,
                                    'F1': np.nan,
                                    'TRAINING[s]': np.nan,
                                    'TESTING[s]': np.nan}, ignore_index=True)

    return scores

def loio(n):
    all_i = range(n)
    for i in all_i:
        test_i = set([i])
        train_i = set(all_i)-test_i
        yield train_i, test_i

def logical_threshold(df, eliminate=True, nan=False):
    with open('./cached/bound.pkl', 'rb') as f:
        bound = pickle.load(f)
    for l in df.columns:
        if l in bound.index:
            lb = df[l] < bound.loc[l, 'lb']
            ub = df[l] > bound.loc[l, 'ub']
            if eliminate:
                bc = ub | lb
                df = df[~bc]
            else:
                if nan:
                    df[l][lb] = np.NaN
                    df[l][ub] = np.NaN
                else:
                    df[l][lb] = bound.loc[l, 'lb']
                    df[l][ub] = bound.loc[l, 'ub']
    return df

def Preprocessing(args, vars, data_path):
    real_instances = pd.DataFrame(class_and_file_generator(data_path,real=True,simulated=False,drawn=False),columns=['class_code', 'instance_path'])
    real_instances = real_instances.loc[real_instances.iloc[:,0].isin(args.abnormal_classes_codes)].reset_index(drop=True)
    sim_instances = pd.DataFrame(class_and_file_generator(data_path,real=False,simulated=True,drawn=False),columns=['class_code', 'instance_path'])
    sim_instances = sim_instances.loc[sim_instances.iloc[:,0].isin(args.abnormal_classes_codes)].reset_index(drop=True)
    drawn_instances = pd.DataFrame(class_and_file_generator(data_path,real=False,simulated=False,drawn=True),columns=['class_code', 'instance_path'])
    drawn_instances = drawn_instances.loc[drawn_instances.iloc[:,0].isin(args.abnormal_classes_codes)].reset_index(drop=True)
    instances = pd.concat([real_instances, sim_instances, drawn_instances]).reset_index(drop=True)
    print(f"number of total instances = {len(real_instances)+ len(sim_instances) + len(drawn_instances)}")
    print(f"number of real instances = {len(real_instances)}")
    print(f"number of simulated instances = {len(sim_instances)}")
    print(f"number of drawn instances = {len(drawn_instances)}")

    # Loads all real, simulated and drawn instances and applies downsample
    instance_id = 0
    df_real_instances, instance_id  = load_and_downsample_instances(real_instances, args.downsample_rate, 'real', instance_id,args)
    df_simul_instances, instance_id = load_and_downsample_instances(sim_instances,args.downsample_rate,'simulated', instance_id,args)
    df_drawn_instances, instance_id = load_and_downsample_instances(drawn_instances,args.downsample_rate,'drawn',instance_id, args)
    df_instances = pd.concat([df_real_instances, df_simul_instances, df_drawn_instances])

    idxs = (df_instances['source']=='real') & (df_instances['instance_id'])
    good_vars = np.isnan(df_instances.loc[idxs][vars]).mean(0) <= args.max_nan_percent
    good_vars = list(good_vars.index[good_vars])
    bad_vars = list(set(vars)-set(good_vars))
    df_instances_good_vars = df_instances.drop(columns=bad_vars, errors='ignore')

    # df_instances_logical = logical_threshold(df_instances_good_vars)
    df_instances_logical = logical_threshold(df_instances_good_vars, eliminate=False)

    # Totally eliminated
    elim_set_total = set(df_instances_good_vars['instance_id']) - set(df_instances_logical['instance_id'])
    # Partially eliminated
    loss = {}
    for id in sorted(list(set(df_instances_good_vars['instance_id']).intersection(set(df_instances_logical['instance_id'])))):
        before =len(df_instances_good_vars[df_instances_good_vars['instance_id']==id])
        after =len(df_instances_logical[df_instances_logical['instance_id']==id])
        if before != after:
            loss[id] = before - after
    elim_set_partial = set(loss.keys())
    for var in list(elim_set_partial):
        df_instances_logical = df_instances_logical[~(df_instances_logical['instance_id'] == var)]
    elim_set = (elim_set_total | elim_set_partial)
    elim_id = sorted(list(elim_set))
    elim_instances = instances.loc[elim_id]

    logical_set = (set(real_instances.index) - elim_set)
    #clean_set = (set(instances.index) - elim_set)
    logical_id = sorted(list(logical_set))
    logical_instances = instances.loc[logical_id]

    frozen = np.zeros(5)
    for id in logical_id:
        scaler = MinMaxScaler()
        idxs = (df_instances_logical['instance_id']==id)
        scaler.fit(df_instances_logical[idxs][good_vars])
        df_scaled = scaler.fit_transform(df_instances_logical[idxs][good_vars])
        df_scaled = pd.DataFrame(df_scaled, columns=good_vars)
        # frozen += np.nanstd(df_scaled[good_vars], 0) < args.std_vars_min
        frozen += ((np.nanstd(df_scaled[good_vars], 0) < args.std_vars_min).astype(bool) + (df_scaled[good_vars].isna().sum()/len(df_scaled) > args.max_frozen_percent).astype(bool))
    clean_vars= np.array(good_vars)
    clean_vars = list(clean_vars[(frozen / len(logical_id) <= args.max_frozen_percent)])
    frozen_vars = list(set(good_vars) - set(clean_vars))
    df_instances_clean_frozen = df_instances_logical.drop(columns=frozen_vars, errors='ignore')
    print(f'\nClean Variables: {clean_vars}')

    eliminate = False
    elim_clean_id = []
    in_frozen = 0
    in_total_frozen = 0
    for id in logical_id:
        scaler = MinMaxScaler()
        idxs = (df_instances_clean_frozen['instance_id']==id)
        df_tmp = df_instances_clean_frozen[idxs][clean_vars]
        scaler.fit(df_tmp)
        df_scaled = scaler.fit_transform(df_tmp)
        df_scaled = pd.DataFrame(df_scaled, columns=clean_vars)
        crit = (np.nanstd(df_scaled[clean_vars], 0) < args.std_vars_min) + (np.sum(np.isnan(df_tmp)) / len(df_tmp) > args.max_frozen_percent).to_numpy()
        if True in crit:
            if eliminate:
                elim_clean_id.append(id)
            else:
                pass
            in_frozen = (df_instances_clean_frozen['instance_id']==id)
            if not eliminate:
                clean_vars_instance = np.array(clean_vars)
                df_instances_clean_frozen.loc[in_frozen.astype(bool),clean_vars_instance[crit]] = 0
                if len(set(clean_vars_instance[crit])) == len(clean_vars):
                    elim_clean_id.append(id)
                    in_total_frozen += (df_instances_clean_frozen['instance_id']==id)

    clean_id = list((set(logical_id) - set(elim_clean_id)))
    if eliminate:
        df_instances_clean = df_instances_clean_frozen[~in_total_frozen.astype(bool)]
    else:
        # if not isinstance(in_total_frozen, int):
        #     df_instances_clean = df_instances_clean_frozen[~in_total_frozen.astype(bool)]
        # else:
            df_instances_clean = df_instances_clean_frozen

    df_instances_clean[clean_vars] = df_instances_clean[clean_vars].interpolate(method='linear')

    return df_instances_clean, clean_vars, instances, real_instances, sim_instances, drawn_instances

def fix_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_data(df, id_set):
    data_crit = 0
    for idx in id_set:
        data_crit +=df['instance_id'] == idx
    return df[data_crit.astype(bool)]

def make_new_data(df, id_set):
    data_crit = 0
    for idx in id_set:
        data_crit +=df['new_instance_id'] == idx
    return df[data_crit.astype(bool)]

def make_trend(df, id_set, input_vars):
    trend =pd.DataFrame()
    for id in id_set:
        slope = df.loc[df['instance_id']==id, input_vars].iloc[1] - df.loc[df['instance_id']==id, input_vars].iloc[0]
        slope[slope >0] = 1  # 상승
        slope[slope==0] = 0  # 동일
        slope[slope <0] = -1 # 하강
        trend = pd.concat([trend, slope], axis=0).reset_index(drop=True)
    return trend

def make_augmentation(df, new_instances, real_instances, sim_instances, drawn_instances):
    for class_num in set(real_instances['class_code']):
        real_idx = list(real_instances[real_instances['class_code'] == class_num].index.values)
        sim_idx = list(sim_instances[sim_instances['class_code'] == class_num].index.values + 1025)
        drawn_idx = list(drawn_instances[drawn_instances['class_code'] == class_num].index.values + 1964)
        syn_idx = np.array(sim_idx + drawn_idx)
        syn_idx_fin = np.array(sim_idx + drawn_idx)
        tot_idx = real_idx + sim_idx + drawn_idx

        df_new = pd.DataFrame()
        df_scale = copy(df)
        for id in tot_idx:
            only = df_scale.loc[df_scale['instance_id'] == id]
            df_new = pd.concat([df_new, only], ignore_index=True)

        if class_num == 2:
            df_new_tmp = copy(df_new)
            for idx in syn_idx:
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'P-MON-CKP'] = 0
                new_id = len(new_instances)
                new_instances.loc[new_id, 'class_code'] = class_num
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'instance_id'] = new_id
                only = df_new_tmp.loc[df_new_tmp['instance_id'] == new_id]
                df_new = pd.concat([df_new, only], ignore_index=True)
                df = pd.concat([df, only], ignore_index=True)
                syn_idx_fin = np.append(syn_idx_fin,new_id)
            df_new_tmp = copy(df_new)

        if class_num == 8:
            df_new_tmp = copy(df_new)
            for idx in syn_idx:
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'P-TPT'] = 0
                new_id = len(new_instances)
                new_instances.loc[new_id, 'class_code'] = class_num
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'instance_id'] = new_id
                only = df_new_tmp.loc[df_new_tmp['instance_id'] == new_id]
                df_new = pd.concat([df_new, only], ignore_index=True)
                df = pd.concat([df, only], ignore_index=True)
                syn_idx_fin = np.append(syn_idx_fin,new_id)

            df_new_tmp = copy(df_new)
            for idx in syn_idx:
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'T-TPT'] = 0
                new_id =len(new_instances)
                new_instances.loc[new_id, 'class_code'] = class_num
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'instance_id'] = new_id
                only = df_new_tmp.loc[df_new_tmp['instance_id'] == new_id]
                df_new = pd.concat([df_new, only], ignore_index=True)
                df = pd.concat([df, only], ignore_index=True)
                syn_idx_fin = np.append(syn_idx_fin,new_id)

            df_new_tmp = copy(df_new)
            for idx in syn_idx:
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'P-TPT'] = 0
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'T-TPT'] = 0
                new_id = len(new_instances)
                new_instances.loc[new_id, 'class_code'] = class_num
                df_new_tmp.loc[df_new_tmp['instance_id'] == idx, 'instance_id'] = new_id
                only = df_new_tmp.loc[df_new_tmp['instance_id'] == new_id]
                df_new = pd.concat([df_new, only], ignore_index=True)
                df = pd.concat([df, only], ignore_index=True)
                syn_idx_fin = np.append(syn_idx_fin,new_id)
    return df, new_instances