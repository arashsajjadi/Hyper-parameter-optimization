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
