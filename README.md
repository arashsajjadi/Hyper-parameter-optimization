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



<pre><code>This is a code block.
</code></pre>
