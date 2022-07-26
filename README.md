# Hyperparameter-optimization
I took an optimization course with my dear professor, Dr. Bijan Ahmadi ([@kakavandi](https://github.com/kakavandi)). For this course, I have to submit two or three projects, which I will share with you in this repository. The focus of these projects is on optimizing the hyperparameters of a neural network model with a powerful package called ray tune.

+ The professor to whom I have to hand over these projects is Dr. Taheri ([@zahta](https://github.com/zahta)), who accepted the effort and accompanied us and Dr. Ahmadi in this course.
+ During these projects I will cooperate with Mr. Mahmoud Hashempour ([@Mahmoud4812](https://github.com/Mahmoud4812)) .

*The data set that I am going to work on during this project is the molecular data set. Since my focus in this project is only on optimizing the hyper-parameters of a neural network, I will refrain from additional explanations about the nature of the data and introduce them briefly.*

<center> 
  
|   |  Dataset |    Task Type   | Tasks | Compunds |      Category     |
|---|:--------:|:--------------:|:-----:|:--------:|:-----------------:|
| 1 |   Bbbp   | Classification |   1   |   2039   |     Physiology    |
| 2 |   HIV    | Classification |   1   |   41127  |     Biophysics    |
| 3 | FreeSolv |   Regression   |   1   |    642   | Pysical Chemistry |
</center>

# hyper-parameter optimization project report

**Arash Sajjadi**

Optimization in data science

Shahid Beheshti University

July 25, 2022

## Abstract

The upcoming project is a hyper-parameter optimization of the neural network, which has been implemented on three datasets of Bbbp, HIV, and FreeSolv. Two datasets describe classification problems, and one dataset that I am working on is regression. Next, I will explain each data set. One of my most essential references during this project will be the ray library article.

## Pre-Introduction

The data set that I am going to work on during this project is the molecular data set. Since my focus in this project is only on optimizing the hyper-parameters of a neural network, I will refrain from additional explanations about the nature of the data and introduce them briefly.

|   | Dataset  | Task Type      | Tasks | Compunds | Category          |
|---|----------|----------------|-------|----------|-------------------|
| 1 | Bbbp     | Classification | 1     | $2,039$  | Physiology        |
| 2 | HIV      | Classification | 1     | $41,127$ | Biophysics        |
| 3 | FreeSolv | Regression     | 1     | $642$    | Pysical Chemistry |

## Introduction

In this section, I would like to put a summary table of the features and labels of the data sets I am dealing with. It is necessary to explain that some features represent a constant value that I have deleted in all datasets after calling in Python. Therefore, I have specified each record's algebraic dimension of the features vector in this table. It should be mentioned that I have partitioned the data set to the training set, validation, and testing in the same proportion.

|          | Number of features | The algebraic dimension of the features | Number of records | Target variable | Training set-Validation set-Test set |
|:--------:|:------------------:|:---------------------------------------:|:-----------------:|:---------------:|:------------------------------------:|
|   Bbbp   |         200        |                   185                   |       2,039       |    { 0 , 1 }    |           70 % -20 % -10 %           |
|    HIV   |         200        |                   192                   |      41 , 127     |    { 0 , 1 }    |           70 % -20 % -10 %           |
| FreeSolv |         200        |                   161                   |        642        |        R        |           70 % -20 % -10 %           |

Also, since I’m presenting all three projects in one report, I’ve highlighted all the code for the **Bbbp** dataset with a light red background, the **HIV** dataset with a light green background, and the **FreeSolv** dataset with a light blue background for more straightforward diagnosis.

---
**NOTE**

Of course, these colors are related to the pdf file of the report. Here, I will mention which code is related to which dataset in the caption section.

---

### Distribution of labels

Examining the distribution of labels is one of the first things we should do when dealing with any data set. I have also plotted this distribution for all three data sets. [^1]
[^1]: Matplotlib library has been used in most visualizations of this report, either directly or indirectly.

![1](https://user-images.githubusercontent.com/47760229/180877694-197b5adb-f457-419d-abb6-22494d00175f.png)
<p>
    <img src=![1](https://user-images.githubusercontent.com/47760229/180877694-197b5adb-f457-419d-abb6-22494d00175f.png) alt>
    <em>Figure 1: Distribution of Bbbp dataset label</em>
</p>

Since there is an imbalance between the class of one and zero labels in Bbbp's dataset (Fig 1), I will partition the data with the same proportion of distribution. [^2]
[^2]: I am not sure that using `stratify` in `train_test_split` does not cause any information leakage. However, this is a standard method, and I will use this method in this project.

![2 (1)](https://user-images.githubusercontent.com/47760229/180879626-ffece4c0-529f-4ba1-9638-a022882e0b4f.png)
<p>
    <img src=![1](https://user-images.githubusercontent.com/47760229/180877694-197b5adb-f457-419d-abb6-22494d00175f.png) alt>
    <em>Figure 2: Distribution of HIV dataset label</em>
</p>

Since there is an imbalance between the class of one and zero labels in HIV's dataset (Fig 2), I will partition the data with the same proportion of distribution.

It is clear that this imbalance is significantly more apparent in the HIV dataset than in the Bbbp dataset.

![3](https://user-images.githubusercontent.com/47760229/180879859-d7ea5af5-6a16-46f0-b383-569b0bdf6675.png)
<p>
    <em>Figure 3: Distribution of FreeSolv dataset labe</em>
</p>

The best way to interpret Figure 3 is to present a frequency table describing this graph. Therefore, I draw your attention to this table.

| count | mean   | std   | min    | 25\%   | 50\%  | 75\%   | max  |
|-------|--------|-------|--------|--------|-------|--------|------|
| 642   | -3.803 | 3.847 | -25.47 | -5.727 | -3.53 | -1.215 | 3.43 |

### Type of the features

Undoubtedly, all the features specify different properties of chemicals. It should be noted that all these data have been normalized by the CDF [^3] scaler method. According to my research, the standard and min-max scaler methods are not applicable for these types of data sets. Also, I should point out that none of the features represent categorical features (or at least I have access to their numerically encoded version).
[^3]:Cumulative Distribution Function

## Preprocessing

Data preprocessing includes Data partitioning, Feature selection, and Missing values subsections that prepare the data for final processing.

### Data partitioning

I have chosen an evaluation method for the best model for these three projects, which I will explain in detail. For a better visual understanding, I drew Figure 4 to make it easier for me to explain what is going to happen. I will use a semi-cross-validation system to train the neural network, which needs to partition the data like Figure 4.

![myImage (1)](https://user-images.githubusercontent.com/47760229/180934541-96d49b82-01ea-4d56-a0ea-f278aa9bedae.png)
<p>
    <em>Figure 4: Data partitioning diagram in all three data sets</em>
</p>




<pre><code>
%%capture
try:
    import ray
except:
    %pip install ray
    import ray

%pip install ray
try:
    import optuna
except:
    %pip install optuna
    import optuna

try:
    from torchmetrics import ConfusionMatrix
except:
    %pip install torchmetrics
    from torchmetrics import ConfusionMatrix
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,  TensorDataset, Dataset
import torch.nn.functional as F
from ray import tune
import os
import ray
from ray.tune.schedulers import ASHAScheduler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
confmat = ConfusionMatrix(num_classes=2)

%%capture
try:
    from featurewiz import featurewiz
except:
    !pip install featurewiz==0.1.70
    from featurewiz import featurewiz
    
#read data
initial_targets=pd.read_csv("bbbp.csv")
initial_features=pd.read_csv("bbbp_global_cdf_rdkit.csv")

initial_features=initial_features.loc[:, (initial_features != initial_features.iloc[0]).any()] 

shuffled_targets=initial_targets.sample(frac=1,random_state=1234).reset_index(drop=True)
shuffled_features=initial_features.sample(frac=1,random_state=1234).reset_index(drop=True)
#shuffling the data to randomize the sequence

X_train, X_test, y_train, y_test = train_test_split(shuffled_features,shuffled_targets["p_np"],test_size=0.1, random_state=1234, stratify=shuffled_targets["p_np"])
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.22, random_state=1234, stratify=y_train)
X_train12 ,X_train34 , y_train12, y_train34=train_test_split(X_train,y_train,test_size=0.5, random_state=1234,stratify=y_train)
X_train1 ,X_train2 , y_train1, y_train2=train_test_split(X_train12,y_train12,test_size=0.5, random_state=1234,stratify=y_train12)
X_train3 ,X_train4 , y_train3, y_train4=train_test_split(X_train34,y_train34,test_size=0.5, random_state=1234,stratify=y_train34)
del X_train12
del X_train34
del y_train12
del y_train34
k_fold_X_train=[X_train1 ,X_train2,X_train3 ,X_train4 ]
k_fold_y_train=[y_train1 ,y_train2,y_train3 ,y_train4 ]
del X_train1
del X_train2
del X_train3
del X_train4
del y_train1
del y_train2
del y_train3
del y_train4

Data=pd.concat([X_train, y_train], axis=1) #reshape data for featurewiz feature selection
target = ['p_np']
from featurewiz import featurewiz 
feature_selection = featurewiz(Data, target, corr_limit=0.97, verbose=2,header=0, nrows=None,outputs="features")

X_train=X_train[feature_selection[0]]
X_valid=X_valid[feature_selection[0]]
X_test=X_test[feature_selection[0]]
y_train=pd.DataFrame(data=y_train)
y_valid=pd.DataFrame(data=y_valid)
y_test=pd.DataFrame(data=y_test)

for i in np.arange(0,4):
  k_fold_X_train[i]=k_fold_X_train[i][feature_selection[0]]
  k_fold_y_train[i]=pd.DataFrame(data=k_fold_y_train[i])
\end{lstlisting}

\begin{lstlisting}[language=Python, caption={import libraries, data reading, train test splitting, feature selection, and deal with missing data} ,backgroundcolor=\color{hiv}]
import ray
import optuna
from torchmetrics import ConfusionMatrix
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,  TensorDataset, Dataset
import torch.nn.functional as F
from ray import tune
import os
import ray
from ray.tune.schedulers import ASHAScheduler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
confmat = ConfusionMatrix(num_classes=2)
from featurewiz import featurewiz
from sklearn.impute import KNNImputer
#read data
initial_targets=pd.read_csv("Downloads/Hiv dataset/hiv.csv")
initial_features=pd.read_csv("Downloads/Hiv dataset/hiv_global_cdf_rdkit.csv")

initial_features=initial_features.loc[:, (initial_features != initial_features.iloc[0]).any()] 

shuffled_targets=initial_targets.sample(frac=1,random_state=1234).reset_index(drop=True).drop("smiles",axis=1)
shuffled_features=initial_features.sample(frac=1,random_state=1234).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(shuffled_features,shuffled_targets,test_size=0.1, random_state=1234,stratify=shuffled_targets["HIV_active"])
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.22, random_state=1234,stratify=y_train["HIV_active"])

#feature selection
Data=pd.concat([X_train, y_train], axis=1)
target = ['HIV_active']
feature_selection = featurewiz(Data.dropna(), target, corr_limit=0.8, verbose=2,header=0, nrows=None,outputs="features")

#dealing with missings
X_train=X_train[feature_selection[0]]
X_valid=X_valid[feature_selection[0]]
X_test=X_test[feature_selection[0]]
y_train=pd.DataFrame(data=y_train)
y_valid=pd.DataFrame(data=y_valid)
y_test=pd.DataFrame(data=y_test)


imputer = KNNImputer(n_neighbors=3)
X_train=pd.DataFrame(data=imputer.fit_transform(X_train))
X_valid=pd.DataFrame(data=imputer.transform(X_valid))
X_test=pd.DataFrame(data=imputer.transform(X_test))


X_train12 ,X_train34 , y_train12, y_train34=train_test_split(X_train,y_train,test_size=0.5, random_state=1234,stratify=y_train["HIV_active"])
X_train1 ,X_train2 , y_train1, y_train2=train_test_split(X_train12,y_train12,test_size=0.5, random_state=1234,stratify=y_train12["HIV_active"])
X_train3 ,X_train4 , y_train3, y_train4=train_test_split(X_train34,y_train34,test_size=0.5, random_state=1234,stratify=y_train34["HIV_active"])
del X_train12
del X_train34
del y_train12
del y_train34
k_fold_X_train=[X_train1 ,X_train2,X_train3 ,X_train4 ]
k_fold_y_train=[y_train1 ,y_train2,y_train3 ,y_train4 ]
del X_train1
del X_train2
del X_train3
del X_train4
del y_train1
del y_train2
del y_train3
del y_train4

k_fold_y_train[0]=pd.DataFrame(data=k_fold_y_train[0])
k_fold_y_train[1]=pd.DataFrame(data=k_fold_y_train[1])
k_fold_y_train[2]=pd.DataFrame(data=k_fold_y_train[2])
k_fold_y_train[3]=pd.DataFrame(data=k_fold_y_train[3])
</code></pre>
<p>
    <em>Listing 1: import libraries, data reading, train test splitting, adn feature
selection (**Bbbp**)
</em>
</p>
