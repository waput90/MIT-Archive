def PrintResult(sample_index, spam_count, ham_count, radius, final_pred, score, X_test, y_test):
    print(f"KNN Result for Email #{sample_index}")
    print(f"Neighbors: {spam_count} Spam, {ham_count} Ham")
    print(f"Distance to furthest neighbor: {radius:.4f}")
    print(f"Result: {final_pred}")
    print(f"Overall Model Accuracy: {score:.2f}%")