import pandas as pd
import numpy as np
import torch
import os
import random
from imblearn.over_sampling import SMOTE


feature_name = ["project", "ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix",
                "ndev", "age", "nuc", "exp", "rexp", "sexp",
                "membership_1","membership_2","membership_3","membership_4"]

label_name_app = ["project","contains_bug"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def convert_dtype_dataframe_app(df,feature_name,group):
    df['project'] = df['project'].map({
        "aFall": group[0],
        "Alfresco": group[1],
        "androidSync": group[2],
        "androidWalpaper": group[3],
        "anySoftKeyboard": group[4],
        "Apg": group[5],
        "Applozic": group[6],
        "delta_chat": group[7],
        "image": group[8],
        "kiwis": group[9],
        "lottie": group[10],
        "ObservableScrollView": group[11],
        "owncloudandroid": group[12],
        "Pageturner": group[13],
        "reddit": group[14],
        })
    df = df.astype({i: 'float32' for i in feature_name})
    df = df.fillna(df.mean())
    return df

def group_data_by_project(data):
    grouped_data = {}
    for row in data:
        value = row[0]
        if value not in grouped_data:
            grouped_data[value] = []
        grouped_data[value].append(row[1:])
    return grouped_data

def load_data(base_path: str,dataset:str,group,train_file,test_file):
    pkl_test = pd.read_pickle(os.path.join(base_path,dataset, test_file))
    pkl_train = pd.read_pickle(os.path.join(base_path,dataset, train_file))

    pkl_train = convert_dtype_dataframe_app(pkl_train, feature_name,group)
    pkl_test = convert_dtype_dataframe_app(pkl_test, feature_name,group)

    X_train, y_train = pkl_train[feature_name].values, pkl_train[label_name_app].values
    X_test, y_test = pkl_test[feature_name].values, pkl_test[label_name_app].values

    return X_train, y_train, X_test, y_test

def extract_data(data):
    tasks_data = []
    for task_id, data_list in data.items():
        task_data = torch.tensor(data_list).float()
        tasks_data.append(task_data)
    return tasks_data

def mini_batches_update(X, Y, mini_batch_size, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    X = np.array(X)
    Y = np.array(Y)

    Y_pos = np.where(Y[:, 0] == 1.0)[0]
    Y_neg = np.where(Y[:, 0] == 0.0)[0]

    num_pos_samples = len(Y_pos)
    num_neg_samples = len(Y_neg)

    pos_batch_size = min(num_pos_samples, mini_batch_size // 2)
    neg_batch_size = mini_batch_size - pos_batch_size

    for k in range(0, m // mini_batch_size + 1):
        if k * mini_batch_size >= m:
            break

        pos_indexes = np.random.choice(Y_pos, size=pos_batch_size, replace=pos_batch_size > num_pos_samples)
        neg_indexes = np.random.choice(Y_neg, size=neg_batch_size, replace=neg_batch_size > num_neg_samples)

        indexes = np.concatenate((pos_indexes, neg_indexes))

        mini_batch_X, mini_batch_Y = X[indexes], Y[indexes]
        mini_batches.append((torch.tensor(mini_batch_X), torch.tensor(mini_batch_Y)))

    return mini_batches

def mini_batches_update1(X, Y, mini_batch_size, seed=0):

    m = X.shape[0]
    mini_batches = list()

    shuffled_X, shuffled_Y = X, Y

    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i][0] == 1.0]
    Y_neg = [i for i in range(len(Y)) if Y[i][0] == 0.0]

    for k in range(0, m // mini_batch_size + 1):
        pos_indexes = np.random.choice(Y_pos, size=mini_batch_size // 2, replace=True)
        neg_indexes = np.random.choice(Y_neg, size=mini_batch_size // 2, replace=True)
        indexes = sorted(list(pos_indexes) + list(neg_indexes))

        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_update_list(X, Y, mini_batch_size=32, seed=0):
    mini_batches_list = []
    for x,y in zip(X,Y):
        mini_batches = mini_batches_update(x, y, mini_batch_size, seed)
        mini_batches_list.append(mini_batches)
    return mini_batches_list

def mini_batch_fill_random(batches_train,max_len):

    train_batchs = []
    for i in range(max_len):
        mini_train_batch = []
        for batch in batches_train:
            if (i >= len(batch)):
                j = random.randint(0, len(batch) - 1)
                mini_train_batch.append(batch[j])
            else:
                mini_train_batch.append(batch[i])
        train_batchs.append(mini_train_batch)
    return train_batchs

def mini_batch_fill_random_delete(batches_train, max_len):
    train_batchs = []
    for i in range(max_len):
        mini_train_batch = []
        for batch in batches_train:
            if len(batch) > max_len:
                batch = random.sample(batch, max_len)
            if i >= len(batch):
                j = random.randint(0, len(batch) - 1)
                mini_train_batch.append(batch[j])
            else:
                mini_train_batch.append(batch[i])
        train_batchs.append(mini_train_batch)
    return train_batchs

def mini_batch_fill_none(batches_train,max_len):
    train_batchs = []
    for i in range(max_len):
        mini_train_batch = []
        for batch in batches_train:
            if (i >= len(batch)):
                mini_train_batch.append(None)
            else:
                mini_train_batch.append(batch[i])
        train_batchs.append(mini_train_batch)
    return train_batchs

def mini_batch_fill_SMOTE(batches_train,max_len):
    smote = SMOTE(random_state=42)

    train_batchs = []
    for i in range(max_len):

        mini_train_batch = []

        for batch in batches_train:

            if (i >= len(batch)):

                X = np.array([sample[0].numpy() for sample in batch]).reshape(-1, 26)
                y = np.array([sample[1].numpy() for sample in batch]).reshape(-1, 1)

                X_res, y_res = smote.fit_resample(X, y)

                j = random.randint(0, len(X_res) - 1)

                mini_train_batch.append((torch.from_numpy(np.array([X_res[j]])).view(1, -1),
                                         torch.from_numpy(np.array([y_res[j]])).view(1, -1)))

            else:
                mini_train_batch.append(batch[i])

        train_batchs.append(mini_train_batch)

    return train_batchs


def generate_data_batch(x_tasks_train,y_tasks_train,mini_batch_size):
    mini_batch_size = mini_batch_size
    batches_train = mini_batches_update_list(X=x_tasks_train,Y=y_tasks_train,mini_batch_size=mini_batch_size)  # list:21
    max_len = 0
    for batch in batches_train:
        if (max_len < len(batch)):
            max_len = len(batch)

    lengths = [len(sublist) for sublist in batches_train]
    avg_len = int(sum(lengths) / len(lengths))

    train_batchs = mini_batch_fill_random_delete(batches_train, avg_len)

    x_batch_data = []
    y_batch_data = []
    for i in range(avg_len):
        x_batch_list = []
        y_batch_list = []
        x_batch_all = []
        y_batch_all = []
        for j in range(len(batches_train)):
            x_batch, y_batch = train_batchs[i][j]

            x_batch, y_batch = x_batch.float().clone().detach(), y_batch.float().clone().detach()
            x_batch_list.append(x_batch)
            y_batch_list.append(y_batch)
            if len(x_batch_all) == 0:
                x_batch_all = x_batch
                y_batch_all = y_batch
            else:
                x_batch_all = torch.cat([x_batch_all, x_batch], dim=0)
                y_batch_all = torch.cat([y_batch_all, y_batch], dim=0)
        x_batch_list.append(x_batch_all)
        y_batch_list.append(y_batch_all)
        x_batch_data.append(x_batch_list)
        y_batch_data.append(y_batch_list)
    return x_batch_data,y_batch_data

def generate_mini_batches(x_tasks_train, y_tasks_train, max_batch_size, seed=0):
    np.random.seed(seed)
    n_projects = len(x_tasks_train)

    avg_len = int(sum(len(project) for project in x_tasks_train) / len(x_tasks_train))
    num_batches = avg_len // max_batch_size + (avg_len % max_batch_size != 0)

    project_batch_sizes = [len(project) // num_batches + (len(project) % num_batches != 0) for project in x_tasks_train]

    batches_train = mini_batches_update_list_unequal(X=x_tasks_train,Y=y_tasks_train,mini_batch_size=project_batch_sizes,num_batches=num_batches)  # list:21

    x_batch_data = []
    y_batch_data = []
    for i in range(num_batches):
        x_batch_list = []
        y_batch_list = []
        x_batch_all = []
        y_batch_all = []
        for j in range(n_projects):
            x_batch, y_batch = batches_train[j][i]

            x_batch, y_batch = x_batch.float().clone().detach(), y_batch.float().clone().detach()

            x_batch_list.append(x_batch)
            y_batch_list.append(y_batch)
            if len(x_batch_all) == 0:
                x_batch_all = x_batch
                y_batch_all = y_batch
            else:
                x_batch_all = torch.cat([x_batch_all, x_batch], dim=0)
                y_batch_all = torch.cat([y_batch_all, y_batch], dim=0)
        x_batch_list.append(x_batch_all)
        y_batch_list.append(y_batch_all)
        x_batch_data.append(x_batch_list)
        y_batch_data.append(y_batch_list)
    return x_batch_data, y_batch_data

def mini_batches_update_list_unequal(X, Y, mini_batch_size, num_batches,seed=0):
    mini_batches_list = []
    i = 0
    for x,y in zip(X,Y):
        mini_batches = mini_batches_update_unequal(x, y, mini_batch_size[i], num_batches,seed)
        mini_batches_list.append(mini_batches)
        i += 1
    return mini_batches_list

def mini_batches_update_unequal(X, Y, mini_batch_size,num_batches, seed=0):
    np.random.seed(seed)
    mini_batches = []

    X = np.array(X)
    Y = np.array(Y)

    Y_pos = np.where(Y[:, 0] == 1.0)[0]
    Y_neg = np.where(Y[:, 0] == 0.0)[0]

    num_pos_samples = len(Y_pos)
    num_neg_samples = len(Y_neg)

    pos_batch_size = min(num_pos_samples, mini_batch_size // 2)
    neg_batch_size = mini_batch_size - pos_batch_size

    for k in range(num_batches):

        pos_indexes = np.random.choice(Y_pos, size=pos_batch_size, replace=pos_batch_size > num_pos_samples)
        neg_indexes = np.random.choice(Y_neg, size=neg_batch_size, replace=neg_batch_size > num_neg_samples)

        indexes = np.concatenate((pos_indexes, neg_indexes))
        np.random.shuffle(indexes)

        mini_batch_X, mini_batch_Y = X[indexes], Y[indexes]
        mini_batches.append((torch.tensor(mini_batch_X), torch.tensor(mini_batch_Y)))

    return mini_batches



def mini_batches_update_unequal_SMOTE(X, Y, mini_batch_size, num_batches, seed=0):
    np.random.seed(seed)
    mini_batches = []

    X = np.array(X)
    Y = np.array(Y)

    Y_pos = np.where(Y[:, 0] == 1.0)[0]
    Y_neg = np.where(Y[:, 0] == 0.0)[0]

    num_pos_samples = len(Y_pos)

    pos_batch_size = min(num_pos_samples, mini_batch_size // 2)
    neg_batch_size = mini_batch_size - pos_batch_size

    smote = SMOTE()

    for k in range(num_batches):
        pos_indexes = np.random.choice(Y_pos, size=pos_batch_size, replace=False)
        neg_indexes = np.random.choice(Y_neg, size=neg_batch_size, replace=False)

        if len(pos_indexes) < pos_batch_size:
            X_pos_smote, Y_pos_smote = smote.fit_resample(X[Y_pos], Y[Y_pos])
            pos_indexes = np.random.choice(range(len(X_pos_smote)), size=pos_batch_size, replace=False)

        if len(neg_indexes) < neg_batch_size:
            X_neg_smote, Y_neg_smote = smote.fit_resample(X[Y_neg], Y[Y_neg])
            neg_indexes = np.random.choice(range(len(X_neg_smote)), size=neg_batch_size, replace=False)

        indexes = np.concatenate((pos_indexes, neg_indexes))
        np.random.shuffle(indexes)

        mini_batch_X, mini_batch_Y = X[indexes], Y[indexes]
        mini_batches.append((torch.tensor(mini_batch_X), torch.tensor(mini_batch_Y)))

    return mini_batches

