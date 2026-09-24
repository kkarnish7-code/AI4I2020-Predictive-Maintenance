import pickle
import pandas as pd

# Load Random Forest model
with open("random_forest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Load SVM model
with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

# Load Decision Tree model
with open("decision_tree_model.pkl", "rb") as file:
    dt_model = pickle.load(file)

# Input features:
# Air Temperature, Process Temperature,
# Rotational Speed, Torque, Tool Wear

new_data = pd.DataFrame([
    [300.0, 310.0, 1500.0, 40.0, 100.0]
])

# Predictions
rf_prediction = rf_model.predict(new_data)
svm_prediction = svm_model.predict(new_data)
dt_prediction = dt_model.predict(new_data)

# Display results
print("Machine Failure Predictions")
print("---------------------------")

print("Random Forest :", rf_prediction[0])
print("SVM           :", svm_prediction[0])
print("Decision Tree :", dt_prediction[0])

print("\nPrediction Meaning:")
print("0 = No Machine Failure")
print("1 = Machine Failure")
