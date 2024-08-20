import binary_logistic_regression as b
import multinomial_logistic_regression as m
import feed_forward_nn as f
import examples as e
import numpy as np
import sklearn.datasets as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


digits = sk.load_digits()

X = digits.data#/255.0

y = np.zeros((X.shape[0], 10))
y[np.arange(X.shape[0]), digits.target] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

f = f.feed_forward_nn(X_train,y_train,["0","1","2","3","4","5","6","7","8","9"])

f.train()

predictions = np.array([f.predict(x) for x in X_test])
test_accuracy = np.mean(predictions == np.array([str(np.argmax(y)) for y in y_test]))
print(f"Test accuracy: {test_accuracy:.4f}")


# true_labels = np.array([str(np.argmax(y)) for y in y_test])

# # Identify incorrect predictions
# incorrect_indices = np.where(predictions != true_labels)[0]

# # Display the incorrectly classified images
# for i in incorrect_indices:
#     plt.imshow(X_test[i].reshape(8, 8))
#     plt.title(f"True Label: {true_labels[i]}, Predicted: {predictions[i]}")
#     plt.show()


# X,y,encodings = e.multinomial_classification_data_four()
# c=m.multinomial_logistic_regression(X,y, encodings)
# cf=f.feed_forward_nn(X,y, encodings)

# c.train()
# cf.train()
