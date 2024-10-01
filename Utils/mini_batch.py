import numpy as np
import torch
from imblearn.over_sampling import SMOTE

def balanced_batch_loader(x, y, batch_size, num_batches):

    x = np.array(x)
    y = np.array(y)
    pos_samples = [(x[i], y[i]) for i in range(len(y)) if y[i] == 1]
    neg_samples = [(x[i], y[i]) for i in range(len(y)) if y[i] == 0]

    for i in range(0, len(pos_samples), batch_size):
        batch_pos = pos_samples[i:i + batch_size]
        batch_neg = neg_samples[i:i + batch_size]

        batch = batch_pos + batch_neg
        x_batch, y_batch = zip(*batch)

        yield list(x_batch), list(y_batch)

def generate_mini_batches(x_tasks_train, y_tasks_train, max_batch_size, seed=0,is_eval = False):

    np.random.seed(seed)
    n_projects = len(x_tasks_train)

    if is_eval == False:
        avg_len = sum(len(project) for project in x_tasks_train) // n_projects
        num_batches = avg_len // max_batch_size + (avg_len % max_batch_size != 0)
        project_batch_sizes = [calculate_project_batch_size(project, num_batches) for project in x_tasks_train]
        batches_train = [mini_batches_update_unequal(project_x, project_y, batch_size, num_batches, seed)
                         for project_x, project_y, batch_size in zip(x_tasks_train, y_tasks_train, project_batch_sizes)]
    else:
        avg_len = sum(len(project) for project in x_tasks_train) // n_projects
        num_batches = avg_len // max_batch_size + (avg_len % max_batch_size != 0)
        project_batch_sizes = [calculate_project_batch_size(project, num_batches) for project in x_tasks_train]
        batches_train = [balanced_batch_loader(project_x, project_y, batch_size, num_batches)
                         for project_x, project_y, batch_size in zip(x_tasks_train, y_tasks_train, project_batch_sizes)]


    return prepare_batches(batches_train, n_projects, num_batches)

def calculate_project_batch_size(project, num_batches):

    return len(project) // num_batches + (len(project) % num_batches != 0)


def  prepare_batches(batches_train, n_projects, num_batches):
    x_batch_data, y_batch_data = [], []

    for i in range(num_batches):
        x_batch_list, y_batch_list = [], []
        x_batch_all, y_batch_all = None, None

        for j in range(n_projects):
            x_batch, y_batch = batches_train[j][i]
            x_batch, y_batch = x_batch.float(), y_batch.float()

            x_batch_list.append(x_batch)
            y_batch_list.append(y_batch)

            x_batch_all = torch.cat([x_batch_all, x_batch], dim=0) if x_batch_all is not None else x_batch
            y_batch_all = torch.cat([y_batch_all, y_batch], dim=0) if y_batch_all is not None else y_batch

        x_batch_list.append(x_batch_all)
        y_batch_list.append(y_batch_all)
        x_batch_data.append(x_batch_list)
        y_batch_data.append(y_batch_list)

    return x_batch_data, y_batch_data

def mini_batches_update_unequal(X, Y, mini_batch_size, num_batches, seed=0):
    np.random.seed(seed)

    X = np.array(X)
    Y = np.array(Y)

    smote = SMOTE(random_state=seed)#,k_neighbors=4)
    X_res, Y_res = smote.fit_resample(X, Y)

    Y_pos_res = np.where(Y_res == 1.0)[0]
    Y_neg_res = np.where(Y_res == 0.0)[0]

    pos_batch_size = calculate_batch_size(len(Y_pos_res), mini_batch_size)
    neg_batch_size = mini_batch_size - pos_batch_size

    mini_batches = []

    for _ in range(num_batches):
        pos_indexes = sample_indexes(Y_pos_res, pos_batch_size)
        neg_indexes = sample_indexes(Y_neg_res, neg_batch_size)
        indexes = np.concatenate((pos_indexes, neg_indexes))
        np.random.shuffle(indexes)

        mini_batches.append((torch.tensor(X_res[indexes]), torch.tensor(Y_res[indexes]).unsqueeze(1)))

    return mini_batches

def calculate_batch_size(num_samples, mini_batch_size):

    return min(num_samples, mini_batch_size // 2)

def sample_indexes(samples, batch_size):

    return np.random.choice(samples, size=batch_size, replace=batch_size > len(samples))
