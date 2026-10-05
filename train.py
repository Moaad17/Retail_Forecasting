import argparse
import os

import joblib
import mlflow
import mlflow.xgboost

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Chemin vers le fichier processed.parquet"
)

parser.add_argument(
    "--model_dir",
    type=str,
    default="./outputs",
    help="Dossier de sauvegarde du modèle"
)

args = parser.parse_args()


# ============================================================
# 2. Charger les données
# ============================================================

print("=" * 60)
print("CHARGEMENT DES DONNÉES")
print("=" * 60)

print(f"Data path : {args.data}")

df = pd.read_parquet(args.data)

print(f"Shape initiale : {df.shape}")

print("\nColonnes :")
print(df.columns.tolist())


# ============================================================
# 3. Vérifications
# ============================================================

target = "Weekly_Sales"

if target not in df.columns:
    raise ValueError(
        f"La colonne cible '{target}' est absente du dataset."
    )

if "Date" not in df.columns:
    raise ValueError(
        "La colonne 'Date' est nécessaire pour effectuer "
        "un split temporel."
    )


# ============================================================
# 4. Conversion de Date
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Vérifier les dates invalides
invalid_dates = df["Date"].isna().sum()

if invalid_dates > 0:
    print(
        f"\nAttention : {invalid_dates} lignes ont une date invalide."
    )

    df = df.dropna(subset=["Date"])


# ============================================================
# 5. Trier chronologiquement
# ============================================================

print("\n" + "=" * 60)
print("TRI CHRONOLOGIQUE")
print("=" * 60)

# Pour un dataset de ventes par Store/Dept,
# on garde l'ordre temporel global.
df = df.sort_values("Date").reset_index(drop=True)

print("Date minimale :", df["Date"].min())
print("Date maximale :", df["Date"].max())


# ============================================================
# 6. Nettoyage de la cible
# ============================================================

df = df.dropna(subset=[target])


# ============================================================
# 7. Transformation des variables booléennes
# ============================================================

bool_columns = df.select_dtypes(
    include=["bool"]
).columns

for col in bool_columns:
    df[col] = df[col].astype(int)


# ============================================================
# 8. Gestion des variables catégorielles
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Ne pas encoder Date ici
if "Date" in categorical_columns:
    categorical_columns.remove("Date")

if categorical_columns:

    print("\nVariables catégorielles :")
    print(categorical_columns)

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )


# ============================================================
# 9. Création de features temporelles à partir de Date
# ============================================================

print("\n" + "=" * 60)
print("FEATURES TEMPORELLES")
print("=" * 60)

df["year"] = df["Date"].dt.year

df["month"] = df["Date"].dt.month

df["week"] = df["Date"].dt.isocalendar().week.astype(int)

df["day_of_year"] = df["Date"].dt.dayofyear

df["quarter"] = df["Date"].dt.quarter

# La colonne Date brute n'est pas envoyée au modèle
df = df.drop(columns=["Date"])


# ============================================================
# 10. Séparation X / y
# ============================================================

X = df.drop(columns=[target])

y = df[target]


print("\nNombre de variables :", X.shape[1])
print("Nombre de lignes :", len(X))


# ============================================================
# 11. Split temporel 80 / 20
# ============================================================

print("\n" + "=" * 60)
print("SPLIT TEMPOREL")
print("=" * 60)

# IMPORTANT :
# On utilise l'ordre chronologique du dataset.
# Les premières observations = TRAIN
# Les dernières observations = TEST

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()


print("Train :", X_train.shape)
print("Test  :", X_test.shape)

print(
    f"\nProportion train : {len(X_train) / len(X):.2%}"
)

print(
    f"Proportion test  : {len(X_test) / len(X):.2%}"
)


# ============================================================
# 12. Vérification absence de chevauchement temporel
# ============================================================

# On récupère les dates avant de les supprimer
# Pour afficher les bornes temporelles du train/test,
# on les reconstruit à partir du dataset original trié.

df_dates = pd.read_parquet(args.data)

df_dates["Date"] = pd.to_datetime(
    df_dates["Date"],
    errors="coerce"
)

df_dates = (
    df_dates
    .dropna(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)

train_end_date = df_dates.iloc[split_index - 1]["Date"]

test_start_date = df_dates.iloc[split_index]["Date"]

print("\nDernière date du TRAIN :", train_end_date)

print("Première date du TEST :", test_start_date)


if test_start_date < train_end_date:
    raise ValueError(
        "Erreur : chevauchement temporel entre TRAIN et TEST."
    )


# ============================================================
# 13. Vérifier les NaN
# ============================================================

print("\n" + "=" * 60)
print("GESTION DES VALEURS MANQUANTES")
print("=" * 60)

missing_train = X_train.isna().sum().sum()
missing_test = X_test.isna().sum().sum()

print("NaN dans X_train :", missing_train)
print("NaN dans X_test  :", missing_test)


# XGBoost peut gérer les NaN.
# On les conserve volontairement.


# ============================================================
# 14. Modèle XGBoost
# ============================================================

print("\n" + "=" * 60)
print("CRÉATION DU MODÈLE")
print("=" * 60)

model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 15. MLflow
# ============================================================

mlflow.start_run()

mlflow.log_param(
    "model",
    "XGBRegressor"
)

mlflow.log_param(
    "n_estimators",
    300
)

mlflow.log_param(
    "learning_rate",
    0.05
)

mlflow.log_param(
    "max_depth",
    8
)

mlflow.log_param(
    "split_strategy",
    "time_based_80_20"
)

mlflow.log_param(
    "train_size",
    len(X_train)
)

mlflow.log_param(
    "test_size",
    len(X_test)
)

mlflow.log_param(
    "train_end_date",
    str(train_end_date)
)

mlflow.log_param(
    "test_start_date",
    str(test_start_date)
)


# ============================================================
# 16. Entraînement
# ============================================================

print("\n" + "=" * 60)
print("ENTRAÎNEMENT")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("Entraînement terminé.")


# ============================================================
# 17. Prédictions
# ============================================================

print("\n" + "=" * 60)
print("PRÉDICTIONS")
print("=" * 60)

y_pred = model.predict(
    X_test
)


# ============================================================
# 18. Évaluation
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


print("\nRésultats :")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)


# ============================================================
# 19. MLflow metrics
# ============================================================

mlflow.log_metric(
    "mae",
    float(mae)
)

mlflow.log_metric(
    "rmse",
    float(rmse)
)


# ============================================================
# 20. Sauvegarde du modèle
# ============================================================

print("\n" + "=" * 60)
print("SAUVEGARDE DU MODÈLE")
print("=" * 60)

os.makedirs(
    args.model_dir,
    exist_ok=True
)


# Modèle XGBoost
model_path = os.path.join(
    args.model_dir,
    "model.json"
)

model.save_model(
    model_path
)


# ============================================================
# 21. Sauvegarde des features
# ============================================================

features_path = os.path.join(
    args.model_dir,
    "features.txt"
)

with open(
    features_path,
    "w",
    encoding="utf-8"
) as f:

    for feature in X.columns:

        f.write(
            feature + "\n"
        )


print(
    f"Modèle : {model_path}"
)

print(
    f"Features : {features_path}"
)


# ============================================================
# 22. Enregistrement MLflow du modèle
# ============================================================

mlflow.xgboost.log_model(
    model,
    artifact_path="model"
)


# ============================================================
# 23. Fin MLflow
# ============================================================

mlflow.end_run()


# ============================================================
# 24. Résumé final
# ============================================================

print("\n" + "=" * 60)
print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
print("=" * 60)

print(
    f"Train observations : {len(X_train)}"
)

print(
    f"Test observations  : {len(X_test)}"
)

print(
    f"Train jusqu'au     : {train_end_date}"
)

print(
    f"Test à partir du   : {test_start_date}"
)

print(
    f"MAE                : {mae:.4f}"
)

print(
    f"RMSE               : {rmse:.4f}"
)

