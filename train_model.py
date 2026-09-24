import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import pickle


# ==========================================
# 1. Load the Training and Testing Datasets
# ==========================================

train_path = "TRAIN_AI4I2020_PDM_DATASET.xlsx"
test_path = "TEST_AI4I2020_PDM_DATASET.xlsx"

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)


# ==========================================
# 2. Separate Features and Labels
# ==========================================

Xtrain = train_df.iloc[:, 0:5]
ytrain = train_df.iloc[:, -1]

Xtest = test_df.iloc[:, 0:5]
ytest = test_df.iloc[:, -1]


# ==========================================
# 3. Random Forest
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(Xtrain, ytrain)

y_pred_rf = rf_model.predict(Xtest)


# ==========================================
# 4. Support Vector Machine (SVM)
# ==========================================

svm_model = SVC(
    kernel="rbf",
    random_state=42
)

svm_model.fit(Xtrain, ytrain)

y_pred_svm = svm_model.predict(Xtest)


# ==========================================
# 5. Decision Tree
# ==========================================

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(Xtrain, ytrain)

y_pred_dt = dt_model.predict(Xtest)


# ==========================================
# 6. Evaluation Function
# ==========================================

def evaluate_model(y_test, y_pred, model_name):

    print("\n===================================")
    print(model_name)
    print("===================================")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-Score :", f1)


# ==========================================
# 7. Evaluate All Models
# ==========================================

evaluate_model(
    ytest,
    y_pred_rf,
    "Random Forest"
)

evaluate_model(
    ytest,
    y_pred_svm,
    "Support Vector Machine"
)

evaluate_model(
    ytest,
    y_pred_dt,
    "Decision Tree"
)


# ==========================================
# 8. Save Trained Models
# ==========================================

with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

with open("decision_tree_model.pkl", "wb") as file:
    pickle.dump(dt_model, file)


print("\nAll models trained, evaluated and saved successfully.")
