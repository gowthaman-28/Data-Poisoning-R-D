# ============================================================
# 10_deep_learning_nids.py
# Deep Learning based NIDS after TDA-based Data Sanitization
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
import joblib

import tensorflow as tf


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

keras = tf.keras

Sequential = keras.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input
EarlyStopping = keras.callbacks.EarlyStopping


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sanitized"
    / "cleaned_dataset_new.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. CHECK DATASET FILE
# ============================================================

print("\n========================================")
print("CHECKING DATASET")
print("========================================")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

print("Dataset found:")
print(DATA_PATH)


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 6. DISPLAY COLUMNS
# ============================================================

print("\n========================================")
print("DATASET COLUMNS")
print("========================================")

for i, column in enumerate(df.columns):

    print(i, ":", column)


# ============================================================
# 7. CHECK TARGET LABEL
# ============================================================

print("\n========================================")
print("LABEL DISTRIBUTION")
print("========================================")

if "label" not in df.columns:

    raise ValueError(
        "Target column 'label' was not found."
    )

print(df["label"].value_counts())

print("\nLabel meaning:")
print("0 = Normal")
print("1 = Attack")


# ============================================================
# 8. CHECK POISONING INFORMATION
# ============================================================

print("\n========================================")
print("POISONING INFORMATION")
print("========================================")

if "is_poisoned" in df.columns:

    print(df["is_poisoned"].value_counts())

else:

    print("is_poisoned column not found.")


# ============================================================
# 9. CHECK MISSING VALUES
# ============================================================

print("\n========================================")
print("MISSING VALUES")
print("========================================")

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    print("No missing values found.")

else:

    print(missing)


# ============================================================
# 10. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n========================================")
print("FIRST 5 ROWS")
print("========================================")

print(df.head())


# ============================================================
# 11. REMOVE COLUMNS NOT USED FOR NIDS
# ============================================================

print("\n========================================")
print("REMOVING UNUSED COLUMNS")
print("========================================")

columns_to_remove = []

if "attack_cat" in df.columns:

    columns_to_remove.append("attack_cat")

if "is_poisoned" in df.columns:

    columns_to_remove.append("is_poisoned")


df = df.drop(
    columns=columns_to_remove
)

print("Removed columns:")

for column in columns_to_remove:

    print("-", column)

print("\nDataset shape after removal:")

print(df.shape)


# ============================================================
# 12. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n========================================")
print("SEPARATING FEATURES AND TARGET")
print("========================================")

X = df.drop(
    columns=["label"]
)

y = df["label"].astype(int)


print("Number of input columns:", X.shape[1])

print("\nTarget distribution:")

print(y.value_counts())


# ============================================================
# 13. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "proto",
    "service",
    "state"
]


# ============================================================
# 14. VERIFY CATEGORICAL COLUMNS
# ============================================================

missing_categorical = [
    column
    for column in categorical_columns
    if column not in X.columns
]

if missing_categorical:

    raise ValueError(
        "Missing categorical columns: "
        + str(missing_categorical)
    )


print("\n========================================")
print("CATEGORICAL FEATURES")
print("========================================")

for column in categorical_columns:

    print("-", column)


# ============================================================
# 15. IDENTIFY NUMERICAL FEATURES
# ============================================================

numerical_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]


print("\n========================================")
print("NUMERICAL FEATURES")
print("========================================")

print(
    "Number of numerical features:",
    len(numerical_columns)
)


# ============================================================
# 16. TRAIN-TEST SPLIT
# ============================================================

print("\n========================================")
print("TRAIN-TEST SPLIT")
print("========================================")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


print("\nTraining label distribution:")

print(
    y_train.value_counts()
)


print("\nTesting label distribution:")

print(
    y_test.value_counts()
)


# ============================================================
# 17. CREATE PREPROCESSING PIPELINE
# ============================================================

print("\n========================================")
print("CREATING PREPROCESSING PIPELINE")
print("========================================")


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            StandardScaler(),

            numerical_columns
        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            categorical_columns
        )

    ]
)


# ============================================================
# 18. FIT PREPROCESSOR ON TRAINING DATA
# ============================================================

print("\n========================================")
print("PREPROCESSING DATA")
print("========================================")


X_train_processed = (
    preprocessor.fit_transform(X_train)
)


X_test_processed = (
    preprocessor.transform(X_test)
)


# ============================================================
# 19. CONVERT DATA TO FLOAT32
# ============================================================

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

y_train = np.asarray(
    y_train,
    dtype=np.float32
)

y_test = np.asarray(
    y_test,
    dtype=np.float32
)


print("\n========================================")
print("PROCESSED DATA")
print("========================================")

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 20. BUILD DEEP LEARNING MODEL
# ============================================================

print("\n========================================")
print("BUILDING DEEP LEARNING MODEL")
print("========================================")


input_features = X_train_processed.shape[1]


model = Sequential([

    Input(
        shape=(input_features,)
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )

])


# ============================================================
# 21. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )

    ]

)


# ============================================================
# 22. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\n========================================")
print("MODEL ARCHITECTURE")
print("========================================")

model.summary()


# ============================================================
# 23. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


# ============================================================
# 24. TRAIN DEEP LEARNING MODEL
# ============================================================

print("\n========================================")
print("TRAINING DEEP LEARNING MODEL")
print("========================================")

print("Training started...")


history = model.fit(

    X_train_processed,

    y_train,

    validation_split=0.20,

    epochs=50,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


print("\nTraining completed.")


# ============================================================
# 25. EVALUATE MODEL
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")


test_results = model.evaluate(

    X_test_processed,

    y_test,

    verbose=0

)


print(
    "Test loss:",
    test_results[0]
)

print(
    "Test accuracy:",
    test_results[1]
)

print(
    "Test precision:",
    test_results[2]
)

print(
    "Test recall:",
    test_results[3]
)

print(
    "Test ROC-AUC:",
    test_results[4]
)


# ============================================================
# 26. GENERATE PREDICTIONS
# ============================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")


y_probability = model.predict(

    X_test_processed,

    verbose=0

).ravel()


# Convert probability into class
# >= 0.5 = Attack
# < 0.5  = Normal

y_prediction = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# 27. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 28. PRINT FINAL METRICS
# ============================================================

print("\n========================================")
print("FINAL DNN RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 29. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(

    classification_report(

        y_test,

        y_prediction,

        target_names=[
            "Normal",
            "Attack"
        ],

        digits=4,

        zero_division=0

    )

)


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")


cm = confusion_matrix(

    y_test,

    y_prediction

)


print(cm)


print("\nConfusion Matrix:")

print(
    "                 Predicted"
)

print(
    "              Normal  Attack"
)

print(
    f"Actual Normal  {cm[0][0]:6d}  {cm[0][1]:6d}"
)

print(
    f"Actual Attack  {cm[1][0]:6d}  {cm[1][1]:6d}"
)


# ============================================================
# 31. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"

    ],

    "Value": [

        accuracy,
        precision,
        recall,
        f1,
        roc_auc

    ]

})


metrics_path = (
    RESULTS_DIR
    / "deep_learning_nids_metrics.csv"
)


metrics_df.to_csv(

    metrics_path,

    index=False

)


print("\nMetrics saved to:")

print(metrics_path)


# ============================================================
# 32. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(

    cm,

    index=[
        "Actual_Normal",
        "Actual_Attack"
    ],

    columns=[
        "Predicted_Normal",
        "Predicted_Attack"
    ]

)


confusion_path = (
    RESULTS_DIR
    / "deep_learning_confusion_matrix.csv"
)


confusion_df.to_csv(

    confusion_path

)


print("\nConfusion matrix saved to:")

print(confusion_path)


# ============================================================
# 33. SAVE PREPROCESSOR
# ============================================================

preprocessor_path = (
    ARTIFACTS_DIR
    / "deep_learning_preprocessor.pkl"
)


joblib.dump(

    preprocessor,

    preprocessor_path

)


print("\nPreprocessor saved to:")

print(preprocessor_path)


# ============================================================
# 34. SAVE DEEP LEARNING MODEL
# ============================================================

model_path = (
    ARTIFACTS_DIR
    / "deep_learning_nids_model.keras"
)


model.save(

    model_path

)


print("\nDeep Learning model saved to:")

print(model_path)


# ============================================================
# 35. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)


history_path = (
    RESULTS_DIR
    / "deep_learning_training_history.csv"
)


history_df.to_csv(

    history_path,

    index=False

)


print("\nTraining history saved to:")

print(history_path)


# ============================================================
# 36. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("DEEP LEARNING NIDS COMPLETED")
print("========================================")

print("Pipeline completed successfully:")

print(
    "TDA Sanitization"
)

print(
    "        ↓"
)

print(
    "Cleaned Dataset"
)

print(
    "        ↓"
)

print(
    "Feature Preprocessing"
)

print(
    "        ↓"
)

print(
    "Deep Neural Network"
)

print(
    "        ↓"
)

print(
    "Normal / Attack Classification"
)

print(
    "\nAll outputs have been saved."
)