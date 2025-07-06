import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text, plot_tree

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Housing_Data.csv', index_col=0)

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check the target variable distribution
print("\nTarget variable (prefarea) distribution:")
print(df['prefarea'].value_counts())

# Prepare the data
# Separate features and target
X = df.drop('prefarea', axis=1)
y = df['prefarea']

# Convert categorical variables to numerical
le_dict = {}
categorical_columns = ['driveway', 'recroom', 'fullbase', 'gashw', 'airco']

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Convert target variablegit 
le_target = LabelEncoder()
y = le_target.fit_transform(y)

print("\nFeatures after encoding:")
print(X.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the decision tree
dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Limit depth to avoid overfitting
    min_samples_split=10,
    min_samples_leaf=5
)

dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
target_names = le_target.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the decision tree rules
print("\nDecision Tree Rules:")
tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print(tree_rules)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=target_names,
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title('Decision Tree for Housing Data')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Make sample predictions
print("\nSample Predictions:")
sample_indices = [0, 1, 2]
for idx in sample_indices:
    sample = X_test.iloc[idx:idx+1]
    prediction = dt_classifier.predict(sample)[0]
    probability = dt_classifier.predict_proba(sample)[0]
    actual = y_test[idx]
    
    print(f"\nSample {idx + 1}:")
    print(f"Features: {sample.iloc[0].to_dict()}")
    print(f"Predicted: {target_names[prediction]} (probability: {probability[prediction]:.3f})")
    print(f"Actual: {target_names[actual]}")