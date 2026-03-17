import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle

from result import PrintResult
from main import KNN_Spam_Detection
import config

# override neighboor value to simulate SPAM
config.n_neighbor = 5

knn, X_test, y_test, y_train, n_neighbors, X_train = KNN_Spam_Detection()

# SIMULATION SPAM 
# Reshape ensures the data is in the 2D format scikit-learn expects
sample_index = 0
sample_point = X_test[sample_index].reshape(1, -1)

# Find neighbors
distances, indices = knn.kneighbors(sample_point)
neighbor_labels = y_train.iloc[indices[0]].values

spam_count = sum(neighbor_labels)
ham_count = n_neighbors - spam_count
final_pred = "SPAM" if knn.predict(sample_point)[0] == 1 else "HAM"
hue_labels = ['Spam' if label == 1 else 'Ham' for label in y_test]

# visualization the neighbors plotted
plt.figure(figsize=(10, 7))

# Removed label='Test Data' to fix the TypeError
sns.scatterplot(
    x=X_test[:, 0], 
    y=X_test[:, 1], 
    hue=hue_labels, 
    palette={'Ham': 'blue', 'Spam': 'red'}, 
    alpha=0.2, 
    edgecolor='none'
)

# Highlight Neighbors (Voters)
# We flatten the coordinates to avoid indexing errors
neighbor_coords = X_train[indices[0]] 
# This list comprehension checks each neighbor's label and picks the color
neighbor_colors = ['red' if label == 1 else 'blue' for label in neighbor_labels]
plt.scatter(
    neighbor_coords[:, 0], neighbor_coords[:, 1], 
    color=neighbor_colors, marker='o', s=150, edgecolor='black', zorder=4, label=f'{n_neighbors} Neighbors'
)

# Target Point
# We use [0, 0] and [0, 1] to get exact x and y values
plt.scatter(
    sample_point[0, 0], sample_point[0, 1], 
    color='yellow', marker='*', s=600, edgecolor='black', zorder=5, label=f'Target ({final_pred})'
)

# boundary circle on whats the point created
radius = distances[0][-1]
circle = Circle(
    (sample_point[0, 0], sample_point[0, 1]), radius, 
    color='green', fill=False, linestyle='--', linewidth=2
)

score = knn.score(X_test, y_test) * 100
PrintResult(sample_index, spam_count, ham_count, radius, final_pred, score, X_test, y_test)

plt.gca().add_patch(circle)

# zoom for radiusus to see the neighbors clearly
margin = radius * 3
plt.xlim(sample_point[0, 0] - margin, sample_point[0, 0] + margin)
plt.ylim(sample_point[0, 1] - margin, sample_point[0, 1] + margin)

plt.title(f'KNN Result: {spam_count} Spam vs {ham_count} Ham Neighbors')
plt.legend(title='Type')
plt.show()