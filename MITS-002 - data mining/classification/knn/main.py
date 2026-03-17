import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
