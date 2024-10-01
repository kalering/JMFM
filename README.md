# JMFM

Since the paper has not yet been published, we currently only provide partial information for RQ1 and RQ4. However, we provide the complete code for RQ2, RQ3 in order to confirm our study.

## RQ1

In RQ1, the experiments use the multi-task learning models from the Model/MTL_Model directory. These models are based on the implementation provided by https://github.com/shenweichen/DeepCTR/tree/master/deepctr/models/multitask. In our research, we re-implemented these models in PyTorch and adjusted them to fit our initial data structure, with the corresponding data located in the Datasets/init_data directory. To protect our research before publication, we are currently not providing other additional codes related to RQ1.

## RQ2

In RQ2, we evaluate the performance of the JMFM method on the dataset. The specific code for performing FCM clustering is omitted here; however, the implementation and relevant functions are provided in **FCM.py** located in the **Utils** directory. The data used for the JMFM method is stored in the **Datasets/data** directory, and running **main.py** will reproduce our experiments. The implementation of XGBoost-SMOTE can be found in the **Model/Baseline** directory, which uses the original source code from the authors’ paper.

Note that since the paper is unpublished, we have protected the source code of JMFM, and you need to paste the **pytransform** directory into **lib/python3.8/site-packages** to run the main function.

## RQ3

The data used for RQ3 is located in the **Datasets/mini_data** directory, and the code for reproducing the experiments is the same as in RQ2.It is important to note that when running experiments on smaller datasets, the SMOTE parameter k_neighbors needs to be adjusted to 4. Without this adjustment, the dataset may be too small to generate a class-balanced training set.

## RQ4

The **w/o_MHA** continues to use data from the **Datasets/mini_data** directory, while **w/o_FCM** utilizes the original, unprocessed data from the **Datasets/init_data** directory, which has not undergone FCM processing. All specific codes are not provided here temporarily.

