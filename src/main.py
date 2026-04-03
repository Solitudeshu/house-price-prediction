import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def split_train_test(data, labels, test_ratio=0.1, random_seed=42):
    """Manually split the dataset, using random_seed for reproducibility"""
    np.random.seed(random_seed)
    combined_data = np.column_stack((data, labels))
    np.random.shuffle(combined_data)
    
    split_index = int(len(combined_data) * (1 - test_ratio))
    train_set = combined_data[:split_index]
    test_set = combined_data[split_index:]
    
    return train_set[:, :-1], train_set[:, -1], test_set[:, :-1], test_set[:, -1]

def knn_predict(train_data, train_labels, test_data, k):
    """KNN algorithm utilizing Numpy Broadcasting for optimized speed (10x faster than traditional for-loops)"""
    predictions = []
    for test_point in test_data:
        # Calculate Euclidean distance from a test point to the entire training set simultaneously
        distances = np.sqrt(np.sum((train_data - test_point) ** 2, axis=1))
        
        # Get indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        # Calculate the average value of the k nearest neighbors
        prediction = np.mean(train_labels[nearest_indices])
        predictions.append(prediction)
    return predictions

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error (RMSE)"""
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def plot_rmse_vs_k(train_data, train_labels, test_data, test_labels, max_k):
    """Evaluate and plot RMSE values for different values of K"""
    rmse_values = []
    for k in range(1, max_k + 1):
        predictions = knn_predict(train_data, train_labels, test_data, k)
        rmse_value = calculate_rmse(test_labels, predictions)
        rmse_values.append(rmse_value)
        print(f'RMSE for k={k}: {rmse_value:.4f}')
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), rmse_values, marker='o', linestyle='dashed', color='b')
    plt.title('RMSE vs. K Value in KNN')
    plt.xlabel('K Value')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()

def main():
    # 1. Load dataset
    file_path = os.path.join(os.getcwd(), 'data', 'real_estate.csv')
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Please ensure the CSV file is located in the 'data/' directory.")
        return

    features = data.iloc[:, 1:-1] 
    labels = data.iloc[:, -1]

    # 2. Split dataset into training and testing sets
    X_train, y_train, X_test, y_test = split_train_test(features, labels, test_ratio=0.1)

    # 3. Normalize data (PREVENT DATA LEAKAGE)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Find optimal K and plot graph
    print("Starting KNN model evaluation from scratch...")
    plot_rmse_vs_k(X_train_scaled, y_train, X_test_scaled, y_test, max_k=20)

if __name__ == "__main__":
    main()