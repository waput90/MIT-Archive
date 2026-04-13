import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

def KNN_Spam_Detection():
    n_neighbors = config.n_neighbor
    # load data set from the data set folder
    # source downloaded from 
    # https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data
    df = pd.read_csv('../dataset/emails.csv')

    # cleanup data since email no. and prediction are not features we want to use for KNN
    X = df.drop(['Email No.', 'Prediction'], axis=1)
    y = df['Prediction']

    # scale dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reducing dimensions to 2D for visualization and KNN
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Split the 2D data for the model
    # KNN does not train data it just stores it, so we can directly use for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=config.test_size, random_state=config.random_state)

    # Train KNN on the 2D points
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    
    # for checking most optimal K
    # Perform cross-validation for different values of k
    # TODO:
    # give citation from research according to...
    # how its transform from contextual to numerical data...
    
    # k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    # cv_scores = []

    # # finding most accurate k value for the model using cross validation, this is important to avoid overfitting or underfitting
    # for k in k_values:
    #     knntest = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    #     scores = cross_val_score(knntest, X_scaled, y, cv=10, scoring='accuracy')
    #     cv_scores.append(scores.mean())
    #     print(f"Cross-Validation Accuracy for k={k}: {scores.mean() * 100:.2f}%")

    # # Find the optimal value of k
    # print(f"Optimal number of neighbors (k): {k_values[cv_scores.index(max(cv_scores))]} with Cross-Validation Accuracy: {max(cv_scores) * 100:.2f}%")
    # RESULT: 
    # Cross-Validation Accuracy for k=1: 85.69%
    # Cross-Validation Accuracy for k=3: 83.74%
    # Cross-Validation Accuracy for k=5: 81.38%
    # Cross-Validation Accuracy for k=7: 79.60%
    # Cross-Validation Accuracy for k=9: 77.57%
    # Cross-Validation Accuracy for k=11: 75.91%
    # Cross-Validation Accuracy for k=13: 74.23%
    # Cross-Validation Accuracy for k=15: 72.78%
    # Cross-Validation Accuracy for k=17: 71.23%
    # Cross-Validation Accuracy for k=19: 69.86%


    return knn, X_test, y_test, y_train, n_neighbors, X_train


# only run if main file was added
if (__name__ == "__main__"):
    knn, X_test, y_test, y_train, n_neighbors, X_train = KNN_Spam_Detection()

    print(f"Overall Model Accuracy: {knn.score(X_test, y_test) * 100:.2f}%")
    
    # visualation of the 2D data with KNN decision boundaries   
    plt.figure(figsize=(10, 7))

    # training points are in the background with low alpha, and test points are highlighted with hue based on their true label (Spam or Ham)
    sns.scatterplot(
        x=X_test[:, 0], 
        y=X_test[:, 1], 
        hue=y_test, 
        palette={0: 'blue', 1: 'red'}, 
        alpha=0.6,
        edgecolor='w'
    )

    # in laymans term HAM = good email, SPAM = bad email
    plt.title('KNN Spam Detection')
    plt.xlabel('Principal Component 1 (Word Frequency Patterns)')
    plt.ylabel('Principal Component 2 (Word Frequency Patterns)')
    plt.legend(title='Type', labels=['Ham (Blue)', 'Spam (Red)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
