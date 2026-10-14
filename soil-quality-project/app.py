import sys
import subprocess
import os
import warnings

# -------------------------------------------------
# INSTALL MISSING PACKAGES
# -------------------------------------------------

packages = {
    "flask": "flask",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn"
}

for import_name, package_name in packages.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name
            ]
        )


# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from flask import Flask, render_template, request

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# -------------------------------------------------
# FLASK
# -------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------
# PROJECT PATH
# -------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

dataset_path = os.path.join(
    BASE_DIR,
    "dataset1.csv"
)

df = pd.read_csv(dataset_path)

df.columns = df.columns.str.strip()


# -------------------------------------------------
# FEATURES AND TARGET
# -------------------------------------------------

FEATURES = [
    "N",
    "P",
    "K",
    "pH",
    "EC",
    "OC",
    "S",
    "Zn",
    "Fe",
    "Cu",
    "Mn",
    "B"
]

TARGET = "Output"

X = df[FEATURES]

y = df[TARGET]


# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# DECISION TREE
# -------------------------------------------------

decision_tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

dt_predictions = decision_tree_model.predict(
    X_test
)


# -------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_model.predict(
    X_test
)


# -------------------------------------------------
# METRICS FUNCTION
# -------------------------------------------------

def get_metrics(predictions):

    return {

        "accuracy": round(
            accuracy_score(
                y_test,
                predictions
            ) * 100,
            2
        ),

        "precision": round(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ) * 100,
            2
        ),

        "recall": round(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ) * 100,
            2
        ),

        "f1": round(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ) * 100,
            2
        )
    }


DT_METRICS = get_metrics(
    dt_predictions
)

RF_METRICS = get_metrics(
    rf_predictions
)


# -------------------------------------------------
# LABELS
# -------------------------------------------------

CLASS_NAMES = {
    0: "Low Fertility",
    1: "Medium Fertility",
    2: "High Fertility"
}

CLASS_ICONS = {
    0: "🌱",
    1: "🌿",
    2: "🌳"
}


# -------------------------------------------------
# EDA PLOTS
# -------------------------------------------------

def generate_eda_plots():

    # Fertility distribution
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Output"
    )

    plt.title(
        "Soil Fertility Class Distribution"
    )

    plt.xlabel(
        "Fertility Class"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.xticks(
        [0, 1, 2],
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "fertility.png"
        ),
        dpi=120
    )

    plt.close()


    # Nitrogen
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="N",
        kde=True
    )

    plt.title(
        "Nitrogen Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "nitrogen.png"
        ),
        dpi=120
    )

    plt.close()


    # Phosphorus
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="P",
        kde=True
    )

    plt.title(
        "Phosphorus Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "phosphorus.png"
        ),
        dpi=120
    )

    plt.close()


    # Potassium
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="K",
        kde=True
    )

    plt.title(
        "Potassium Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "potassium.png"
        ),
        dpi=120
    )

    plt.close()


    # Correlation
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        df.corr(
            numeric_only=True
        ),
        annot=True,
        cmap="YlGn",
        fmt=".2f"
    )

    plt.title(
        "Correlation Between Soil Features"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "correlation.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------
# MODEL PLOTS
# -------------------------------------------------

def generate_model_plots():

    # Decision Tree Confusion Matrix
    dt_cm = confusion_matrix(
        y_test,
        dt_predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        dt_cm,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=[
            "Low",
            "Medium",
            "High"
        ],
        yticklabels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    plt.title(
        "Decision Tree Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "dt_confusion.png"
        ),
        dpi=120
    )

    plt.close()


    # Random Forest Confusion Matrix
    rf_cm = confusion_matrix(
        y_test,
        rf_predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=[
            "Low",
            "Medium",
            "High"
        ],
        yticklabels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "rf_confusion.png"
        ),
        dpi=120
    )

    plt.close()


    # Random Forest Feature Importance
    importance = pd.Series(
        random_forest_model.feature_importances_,
        index=FEATURES
    )

    importance = importance.sort_values(
        ascending=False
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        x=importance.values,
        y=importance.index
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Soil Feature"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "feature_importance.png"
        ),
        dpi=120
    )

    plt.close()


    # Decision Tree visualization
    plt.figure(figsize=(20, 10))

    plot_tree(
        decision_tree_model,
        feature_names=FEATURES,
        class_names=[
            "Low",
            "Medium",
            "High"
        ],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7
    )

    plt.title(
        "Decision Tree - First Levels"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "decision_tree.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------
# HOME
# -------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        rows=df.shape[0],
        columns=df.shape[1]
    )


# -------------------------------------------------
# SOIL GUIDE
# -------------------------------------------------

@app.route("/soil-guide")
def soil_guide():

    return render_template(
        "soil_guide.html"
    )


# -------------------------------------------------
# EDA
# -------------------------------------------------

@app.route("/eda")
def eda():

    generate_eda_plots()

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    class_counts = (
        df["Output"]
        .value_counts()
        .sort_index()
    )

    description = (
        df.describe()
        .round(2)
        .to_html(
            classes="data-table",
            border=0
        )
    )

    return render_template(
        "eda.html",

        missing=missing_values,

        duplicates=duplicate_rows,

        low=int(
            class_counts.get(0, 0)
        ),

        medium=int(
            class_counts.get(1, 0)
        ),

        high=int(
            class_counts.get(2, 0)
        ),

        description=description
    )


# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

@app.route("/models")
def models():

    generate_model_plots()

    if (
        RF_METRICS["accuracy"]
        >=
        DT_METRICS["accuracy"]
    ):

        higher_accuracy_model = (
            "Random Forest"
        )

    else:

        higher_accuracy_model = (
            "Decision Tree"
        )

    return render_template(
        "models.html",

        dt=DT_METRICS,

        rf=RF_METRICS,

        train_rows=len(
            X_train
        ),

        test_rows=len(
            X_test
        ),

        higher_accuracy_model=
        higher_accuracy_model
    )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

@app.route(
    "/predict",
    methods=[
        "GET",
        "POST"
    ]
)
def predict():

    prediction = None

    error = None

    form_values = {}

    if request.method == "POST":

        try:

            values = []

            for feature in FEATURES:

                value = float(
                    request.form[
                        feature
                    ]
                )

                values.append(
                    value
                )

                form_values[
                    feature
                ] = value


            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )


            dt_result = int(
                decision_tree_model.predict(
                    input_data
                )[0]
            )


            rf_result = int(
                random_forest_model.predict(
                    input_data
                )[0]
            )


            prediction = {

                "dt_class":
                    dt_result,

                "dt_label":
                    CLASS_NAMES[
                        dt_result
                    ],

                "dt_icon":
                    CLASS_ICONS[
                        dt_result
                    ],

                "rf_class":
                    rf_result,

                "rf_label":
                    CLASS_NAMES[
                        rf_result
                    ],

                "rf_icon":
                    CLASS_ICONS[
                        rf_result
                    ]
            }


        except ValueError:

            error = (
                "Please enter valid numeric values "
                "for every soil parameter."
            )

        except KeyError:

            error = (
                "Please complete every field."
            )


    return render_template(
        "predict.html",

        prediction=prediction,

        error=error,

        values=form_values
    )


# -------------------------------------------------
# START SERVER
# -------------------------------------------------

if __name__ == "__main__":

    print()
    print("----------------------------------")
    print(" SoilSense ML Project")
    print("----------------------------------")
    print("Decision Tree Accuracy:",
          DT_METRICS["accuracy"], "%")

    print("Random Forest Accuracy:",
          RF_METRICS["accuracy"], "%")

    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print("----------------------------------")

    app.run(
        debug=True,
        port=5000
    )