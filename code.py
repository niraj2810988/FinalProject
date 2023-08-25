import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
###data = pd.read_csv('topSubscribed.csv')

data=pd.read_csv('/Users/niraj/Python Programming/Big Data/Final Project/topSubscribed.csv')


# Display the first few rows of the dataset
print(data.head())

# Analyze channel categories distribution
category_distribution = data['Category'].value_counts()

# Plot the distribution
plt.figure(figsize=(12, 6))
category_distribution.plot(kind='bar')
plt.xticks(rotation=90)
plt.xlabel('Channel Category')
plt.ylabel('Number of Channels')
plt.title('Channel Categories Distribution')
plt.tight_layout()
plt.show()


# Encode categorical features (if needed)
label_encoder = LabelEncoder()
data['CategoryEncoded'] = label_encoder.fit_transform(data['Category'])

# Features and target
X = data[['Subscribers', 'Video Views', 'Video Count']]
y = data['CategoryEncoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
k = 5  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# Encode categorical features (if needed)
label_encoder = LabelEncoder()
data['CategoryEncoded'] = label_encoder.fit_transform(data['Category'])

# Features and target
X = data[['Subscribers', 'Video Views', 'Video Count']]
y = data['CategoryEncoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
k = 5  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test)

# Create a confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion Matrix:\n", confusion)

