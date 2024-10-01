import numpy as np
from fcmeans import FCM
import pandas as pd
import os
import torch
from sklearn.preprocessing import StandardScaler
from collections import Counter

feature_name = ["nd", "nf", "la", "ld", "ndev","nuc","contains_bug"]
def convert_dtype_dataframe_app(df,feature_name):
    df = df.astype({i: 'float32' for i in feature_name})
    mean_values = df.drop(['project', 'contains_bug'], axis=1).mean()
    df.fillna(mean_values, inplace=True)
    return df

def group_data_by_project(df):
    grouped_data = {}
    for project_id, group in df.groupby('project'):
        grouped_data[project_id] = group.drop('project', axis=1).values
    return grouped_data

def scaler_data(group_data):
    for key, value in group_data.items():
        scaler = StandardScaler().fit(group_data[key])
        group_data[key] = scaler.transform(value)
    return group_data

def gaussian_similarity(center1, center2, sigma1, sigma2):
    exponent = -(center1 - center2) ** 2 / (sigma1 + sigma2) ** 2
    similarity = torch.exp(exponent)
    similarity = torch.clamp(similarity, max=1.0)
    return similarity


def fuzzy_similarity(centers1, centers2,memberships1,memberships2):
    memberships1 = torch.tensor(memberships1) if not isinstance(memberships1, torch.Tensor) else memberships1
    memberships2 = torch.tensor(memberships2) if not isinstance(memberships2, torch.Tensor) else memberships2
    centers1 = torch.tensor(centers1) if not isinstance(centers1, torch.Tensor) else centers1
    centers2 = torch.tensor(centers2) if not isinstance(centers2, torch.Tensor) else centers2
    similarities = []
    for i in range(centers1.size(0)):
        sigma1 = torch.std(memberships1[:, i])
        sigma2 = torch.std(memberships2[:, i])
        sim = gaussian_similarity(centers1[i], centers2[i], sigma1, sigma2)
        similarities.append(sim)
    return torch.mean(torch.stack(similarities)).item()


def similarity_matrix_calculate(project_centers, project_memberships):
    num_projects = len(project_centers)
    similarity_matrix = np.zeros((num_projects, num_projects))

    project_keys = list(project_centers.keys())
    for i in range(num_projects):
        for j in range(i, num_projects):
            if i != j:
                project1 = project_keys[i]
                project2 = project_keys[j]
                sim = fuzzy_similarity(project_centers[project1], project_centers[project2],
                                       project_memberships[project1], project_memberships[project2])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            else:
                similarity_matrix[i, j] = 1
    return similarity_matrix

def normalize_and_cluster_data(grouped_data,n_clusters=3):
    scaled_data = []
    project_indices = []
    offset = 0
    project_slices = {}

    for key, data in grouped_data.items():
        scaler = StandardScaler()
        scaled_project_data = scaler.fit_transform(data)
        scaled_data.append(scaled_project_data)
        project_indices.extend([key] * len(scaled_project_data))
        project_slices[key] = (offset, offset + len(scaled_project_data))
        offset += len(scaled_project_data)

    combined_data = np.vstack(scaled_data)

    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(combined_data)

    project_centers = {}
    project_memberships = {}
    for project_id, (start_idx, end_idx) in project_slices.items():
        project_centers[project_id] = fcm.centers
        project_memberships[project_id] = fcm.u[start_idx:end_idx, :]

    return project_centers, project_memberships


def get_memberships(n_clusters,file=None,data=None):
    if file:
        root_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(root_path,"data","app")
        data = pd.read_pickle(os.path.join(data_path,file))
    else:
        data = data

    data = convert_dtype_dataframe_app(data, feature_name)
    group_data = group_data_by_project(data)
    project_centers, project_memberships = normalize_and_cluster_data(group_data,n_clusters)

    return project_centers, project_memberships

def getgroup(n_clusters=3,file=None,data=None):
    project_centers, project_memberships = get_memberships(n_clusters,file,data)
    group = []
    for key in project_memberships:
        primary_cluster = np.argmax(project_memberships[key], axis=1)
        counter = Counter(primary_cluster)
        for number, count in counter.items():
            print(f"project {key} have {count} samples belong to {number} class")
        most_common_num, most_common_count = counter.most_common(1)[0]
        print(f"Project {key} primary cluster: {most_common_num},samples is {most_common_count}/{len(primary_cluster)}")
        group.append(most_common_num)
    return group

