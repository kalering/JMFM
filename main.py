from sklearn.preprocessing import StandardScaler
from torch import optim
import torch
import torch.nn as nn
from sklearn.metrics import *
import warnings
import time
import numpy as np

from Model.JMFM import JMFM
from Utils import mini_batch, functions

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MTL_eval(x_tasks_test, y_tasks_test, model):

    with torch.no_grad():
        x_tasks_test = [x.to(device) for x in x_tasks_test]

        y_pred_prob = model(x_tasks_test)
        y_pred_labels = [(y >= 0.5).float().cpu().numpy() for y in y_pred_prob]
        y_pred_prob_cpu = [np.array(labels.cpu()) for labels in y_pred_prob]
        y_true_labels = [np.array(labels) for labels in y_tasks_test]

        accuracy = np.mean([accuracy_score(y_true, y_pred) for y_true, y_pred in zip(y_true_labels, y_pred_labels)])
        precision = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(y_true_labels, y_pred_labels)])
        recall = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(y_true_labels, y_pred_labels)])
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        f1_scores = [f1_score(y_true, y_pred) for y_true, y_pred in zip(y_true_labels, y_pred_labels)]
        mcc_scores = [matthews_corrcoef(y_true, y_pred) for y_true, y_pred in zip(y_true_labels, y_pred_labels)]
        auc_scores = [roc_auc_score(y_true, y_pred) for y_true, y_pred in zip(y_true_labels, y_pred_prob_cpu)]
        print("F1 Scores:", f1_scores)
        print("MCC Scores:", mcc_scores)
        print("AUC Scores:", auc_scores)

        avg_f1 = np.mean(f1_scores)
        avg_mcc = np.mean(mcc_scores)
        avg_auc = np.mean(auc_scores)
        print(f"Average F1: {avg_f1}")
        print(f"Average MCC: {avg_mcc}")
        print(f"Average AUC: {avg_auc}")


def MTL_train(x_tasks_train, y_tasks_train, model, optimizer, loss_fn, num_epochs, mini_batch_size, task_weights):
    print("training model")
    steps = 0

    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        num_batches = 0
        model.train()
        x_batch_list, y_batch_list = mini_batch.generate_mini_batches(x_tasks_train, y_tasks_train,
                                                                       max_batch_size=mini_batch_size, seed=42)

        for x_batch_data, y_batch_data in zip(x_batch_list, y_batch_list):
            x_batch_data = [x.to(device) for x in x_batch_data]
            y_batch_data = [y.to(device) for y in y_batch_data]
            optimizer.zero_grad()
            y_hats = model(x_batch_data)

            batch_loss = sum(loss_fn(y_hat, y_task) * weight
                             for y_hat, y_task, weight in zip(y_hats, y_batch_data, task_weights))
            total_loss += batch_loss.item()
            num_batches += 1

            batch_loss.backward()

            optimizer.step()
            steps += 1
            if steps % 10 == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, batch_loss.item()))
                with torch.no_grad():
                    f1_scores = [f1_score(y_task.cpu(), torch.round(y_hat).cpu()) for y_hat, y_task in
                                 zip(y_hats, y_batch_data)]
                    batch_f1_scores = sum(f1_scores) / len(f1_scores)
                    print(f"Current step average F1 is：{batch_f1_scores}")
                    auc_scores = [roc_auc_score(y_task.cpu(), y_hat.cpu()) for y_hat, y_task in
                                  zip(y_hats, y_batch_data)]
                    batch_auc_scores = sum(auc_scores) / len(auc_scores)
                    print(f"Current step average AUC is：{batch_auc_scores}")
                    del y_hats, y_batch_data
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
    print("---------------------Finishing the training PLE model---------------------")


if __name__ == "__main__":
    start_time = time.time()
    print("Running model")
    base_path = "Datasets/"
    dataset = "data/"
    train_file, test_file = "data_train.pkl", "data_test.pkl"

    functions.set_seed(seed=42)
    group = [i for i in range(15)]
    X_train, y_train, X_test, y_test = functions.load_data(base_path, dataset, group,train_file,test_file)

    X_grouped_train = functions.group_data_by_project(X_train)
    X_grouped_test = functions.group_data_by_project(X_test)
    y_grouped_train = functions.group_data_by_project(y_train)
    y_grouped_test = functions.group_data_by_project(y_test)

    num_probability_features = 4
    for key, value in X_grouped_train.items():
        for key, value in X_grouped_train.items():
            value = np.array(value)
            non_prob_features_train = value[:, :-num_probability_features]
            prob_features_train = value[:, -num_probability_features:]

            X_grouped_test[key] = np.array(X_grouped_test[key])
            non_prob_features_test = X_grouped_test[key][:, :-num_probability_features]
            prob_features_test = X_grouped_test[key][:, -num_probability_features:]

            scaler = StandardScaler().fit(non_prob_features_train)
            standardized_non_prob_features_train = scaler.transform(non_prob_features_train)
            standardized_non_prob_features_test = scaler.transform(non_prob_features_test)

            X_grouped_train[key] = np.hstack((standardized_non_prob_features_train, prob_features_train))
            X_grouped_test[key] = np.hstack((standardized_non_prob_features_test, prob_features_test))

    x_tasks_train = functions.extract_data(X_grouped_train)
    y_tasks_train = functions.extract_data(y_grouped_train)
    y_tasks_test = functions.extract_data(y_grouped_test)
    x_tasks_test = []
    x_tasks_test_all = []
    for task_id, data_list in X_grouped_test.items():
        task_data = torch.tensor(data_list).float()
        x_tasks_test.append(task_data)
        if len(x_tasks_test_all) == 0:
            x_tasks_test_all = task_data
        else:
            x_tasks_test_all = torch.cat([x_tasks_test_all, task_data], dim=0)
    x_tasks_test.append(x_tasks_test_all)

    print("building model")

    first_key = list(X_grouped_train.keys())[0]
    first_value = X_grouped_train[first_key]
    first_array = first_value[0]
    shape = first_array.shape
    input_size = shape[0]
    num_tasks = len(X_grouped_train)

    num_specific_experts = 80
    num_shared_experts = 258
    experts_out = [106, 268]
    experts_hidden = 146
    towers_hidden = 191
    num_epochs = 5
    lr = 0.001
    level = len(experts_out)
    mini_batch_size = 32

    model = JMFM(input_size=input_size, num_specific_experts=num_specific_experts,
                    num_shared_experts=num_shared_experts, experts_out=experts_out, experts_hidden=experts_hidden,
                    towers_hidden=towers_hidden, num_tasks=num_tasks, level=level).to(device)


    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    task_data_counts = [len(task_data) for task_data in x_tasks_train]
    total_data_count = sum(task_data_counts)
    task_weights = [(count / total_data_count) for count in task_data_counts]

    MTL_train(x_tasks_train, y_tasks_train, model, optimizer, loss_fn, num_epochs, mini_batch_size, task_weights)
    MTL_eval(x_tasks_test, y_tasks_test, model)

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print("running time：{} hours {} minutes {:.2f} seconds".format(hours, minutes, seconds))
