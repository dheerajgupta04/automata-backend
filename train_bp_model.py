import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


DATASET_PATH = "bp_windowed_dataset.csv"


def main():

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(DATASET_PATH)

    print("Dataset shape:", df.shape)

    # -----------------------------
    # Features and Targets
    # -----------------------------
    feature_cols = ["heart_rate", "ptt", "amplitude", "ac_dc_ratio"]

    X = df[feature_cols].values
    y_sbp = df["systolic"].values
    y_dbp = df["diastolic"].values

    # -----------------------------
    # Standardize Features
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Polynomial Features
    # -----------------------------
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # -----------------------------
    # 5-Fold Cross Validation
    # -----------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    sbp_mae_list = []
    dbp_mae_list = []

    for train_index, test_index in kf.split(X_poly):

        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train_sbp, y_test_sbp = y_sbp[train_index], y_sbp[test_index]
        y_train_dbp, y_test_dbp = y_dbp[train_index], y_dbp[test_index]

        sbp_model = Ridge(alpha=1.0)
        dbp_model = Ridge(alpha=1.0)

        sbp_model.fit(X_train, y_train_sbp)
        dbp_model.fit(X_train, y_train_dbp)

        sbp_pred = sbp_model.predict(X_test)
        dbp_pred = dbp_model.predict(X_test)

        sbp_mae_list.append(mean_absolute_error(y_test_sbp, sbp_pred))
        dbp_mae_list.append(mean_absolute_error(y_test_dbp, dbp_pred))

    print("\n===== Polynomial Ridge Results =====")
    print("SBP MAE:", round(np.mean(sbp_mae_list), 2), "mmHg")
    print("DBP MAE:", round(np.mean(dbp_mae_list), 2), "mmHg")

    print("\nStd Deviation:")
    print("SBP STD:", round(np.std(sbp_mae_list), 2))
    print("DBP STD:", round(np.std(dbp_mae_list), 2))

    # -----------------------------
    # Train Final Model On Full Dataset
    # -----------------------------
    print("\nTraining final models on full dataset...")

    final_sbp_model = Ridge(alpha=1.0)
    final_dbp_model = Ridge(alpha=1.0)

    final_sbp_model.fit(X_poly, y_sbp)
    final_dbp_model.fit(X_poly, y_dbp)

    # -----------------------------
    # Save Models
    # -----------------------------
    joblib.dump(final_sbp_model, "sbp_model.pkl")
    joblib.dump(final_dbp_model, "dbp_model.pkl")
    joblib.dump(scaler, "bp_scaler.pkl")
    joblib.dump(poly, "bp_poly.pkl")

    print("\nModels saved:")
    print(" - sbp_model.pkl")
    print(" - dbp_model.pkl")
    print(" - bp_scaler.pkl")
    print(" - bp_poly.pkl")


if __name__ == "__main__":
    main()