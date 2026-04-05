# Real Estate Price Prediction

## Introduction

The Real Estate Price Prediction project is a machine learning implementation designed to estimate house prices based on various historical and geographical features, such as house age, distance to the nearest MRT station, and the number of nearby convenience stores.

Unlike standard implementations that rely entirely on high-level library functions, this project emphasizes a deep understanding of core machine learning mechanics by implementing the K-Nearest Neighbors (KNN) algorithm entirely from scratch. Utilizing vectorized NumPy operations, this approach ensures highly optimized execution speed while demonstrating fundamental mathematical and programming principles.

## Features

* **Custom KNN Algorithm:** A from-scratch implementation of the K-Nearest Neighbors regression model utilizing NumPy broadcasting for computation speed optimization (outperforming traditional for-loop iterations).
* **Data Normalization:** Robust data scaling using Min-Max normalization, with strict separation between training and testing sets to prevent data leakage.
* **Performance Evaluation:** Automated calculation of the Root Mean Squared Error (RMSE) across various 'K' values to identify the optimal hyperparameter.
* **Data Visualization:** Built-in capabilities to visualize the relationship between K-values and model error rates via Matplotlib.

## Tech Stack

* **Language:** Python 3
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Preprocessing:** Scikit-learn (MinMaxScaler)

## Project Structure

```text
House-Price-Prediction/
|-- data/
|   |-- real_estate.csv                         
|-- src/
|   |-- main.py                   
|-- .gitignore                    
|-- requirements.txt              
|-- README.md                     
```

## Setup and Usage

**Prerequisites:**
* Python 3.8 or higher installed on your system.

**Installation:**
Clone the repository to your local machine and install the required dependencies:

```bash
pip install -r requirements.txt
```

**Execution:**
Run the main script to start the data processing and evaluation process:

```bash
python src/main.py
```

## Methodology

1. **Data Preparation:** The dataset is shuffled and split manually into training and testing sets (default 90/10 split) using a fixed random seed to ensure reproducibility.
2. **Scaling:** Features are normalized to a [0, 1] range. The scaler is fitted exclusively on the training data to maintain the integrity of the test set.
3. **Prediction:** The custom KNN algorithm calculates the Euclidean distance between test points and the training set simultaneously using matrix operations. It then averages the target values of the 'K' nearest neighbors to output a prediction.

## Authors

**Course:** CSC00004 - Introduction to Information Technology

**Institution:** University of Science, VNU-HCM  

**Faculty:** Faculty of Information Technology (FIT)  

**Team Members**
* Phan Minh Anh - 24120498
* Ngo Nguyen Dong Quan - 24120505 
* Ho Ngoc Lan Anh - 24120256
* Phan Thi Ngoc Khanh - 24120071
* Huynh Hoang Yen - 24120246
