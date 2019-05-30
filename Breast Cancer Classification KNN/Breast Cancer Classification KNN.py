from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.target)
#print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,breast_cancer_data.target, test_size = 0.2, random_state = 100)
max = 0
maxScore = 0
k_list = range(1,101)
accuracies = []
for i in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(training_data, training_labels)
    score = classifier.score(validation_data, validation_labels)
    accuracies.append(score)
    if score > maxScore:
        maxScore = score
        max = i

print("K:", max, "\nAccuracy:", maxScore)
plt.plot(k_list, accuracies)
plt.xlabel("K")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifer Accuracy")
plt.show()

classifier = KNeighborsClassifier(n_neighbors = max)
classifier.fit(training_data, training_labels)
score = classifier.score(validation_data, validation_labels)

