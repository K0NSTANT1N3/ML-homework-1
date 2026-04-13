from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class HousePreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = X.copy()

        # 1. define nominal columns
        # must be before categorical_cols so we can exclude them
        self.nominal_cols = [
            "MSZoning", "LotConfig", "LandContour", "Neighborhood",
            "Condition1", "Condition2", "BldgType", "HouseStyle",
            "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
            "MasVnrType", "Foundation", "Heating", "CentralAir",
            "Electrical", "GarageType", "SaleType", "SaleCondition",
            "MSSubClass", "MoSold", "YrSold"
        ]
        self.nominal_cols = [c for c in self.nominal_cols if c in X.columns]

        # 2. learn medians for numerics
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.numeric_medians = X[self.numeric_cols].median()

        # 3.learn categories
        # exclude nominal_cols — they get OHE, not mode-filled
        all_str_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
        self.categorical_cols = [
            c for c in all_str_cols if c not in self.nominal_cols
        ]
        if self.categorical_cols:
            self.categorical_modes = X[self.categorical_cols].mode().iloc[0]
        else:
            self.categorical_modes = {}

        # 4. learn ohe categories
        # learn from X_train only — prevents unseen categories
        self.ohe_categories_ = {}
        for col in self.nominal_cols:
            self.ohe_categories_[col] = X[col].dropna().unique().tolist()

        return self

    def transform(self, X):
        X = X.copy()

        # 1. fill NA's meaning none
        # NA here means house simply doesn't have that feature
        none_fill_cols = [
            "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "MasVnrType"
        ]
        for col in none_fill_cols:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        # 2. fill NA's meaning zero
        # NA here means no garage/basement = 0 area, 0 cars etc.
        zero_fill_cols = [
            "GarageYrBlt", "MasVnrArea",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
            "BsmtFullBath", "BsmtHalfBath"
        ]
        for col in zero_fill_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # ── STEP 3: FILL LOTFRONTAGE SMARTLY ─────────────
        # use neighborhood median — similar houses have similar lot size
        if "LotFrontage" in X.columns:
            X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median())
            )
            # fallback: if whole neighborhood is missing, use global median
            X["LotFrontage"] = X["LotFrontage"].fillna(
                self.numeric_medians["LotFrontage"]
            )

        # ── STEP 4: FILL REMAINING NUMERIC NAs WITH MEDIAN
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.numeric_medians[col])

        # ── STEP 5: FILL REMAINING CATEGORICAL NAs WITH MODE
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_modes[col])

        # ── STEP 6: ORDINAL ENCODING ──────────────────────
        # these columns have a natural order (Po < Fa < TA < Gd < Ex)
        # so we encode them as numbers preserving that order
        quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        quality_cols = [
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
            "HeatingQC", "KitchenQual", "FireplaceQu",
            "GarageQual", "GarageCond"
        ]
        for col in quality_cols:
            if col in X.columns:
                X[col] = X[col].map(quality_map)

        if "BsmtExposure" in X.columns:
            X["BsmtExposure"] = X["BsmtExposure"].map(
                {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
            )
        if "BsmtFinType1" in X.columns:
            X["BsmtFinType1"] = X["BsmtFinType1"].map(
                {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
            )
        if "BsmtFinType2" in X.columns:
            X["BsmtFinType2"] = X["BsmtFinType2"].map(
                {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
            )
        if "GarageFinish" in X.columns:
            X["GarageFinish"] = X["GarageFinish"].map(
                {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
            )
        if "Functional" in X.columns:
            X["Functional"] = X["Functional"].map(
                {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
                 "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}
            )
        if "LotShape" in X.columns:
            X["LotShape"] = X["LotShape"].map(
                {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
            )
        if "LandSlope" in X.columns:
            X["LandSlope"] = X["LandSlope"].map(
                {"Sev": 1, "Mod": 2, "Gtl": 3}
            )
        if "PavedDrive" in X.columns:
            X["PavedDrive"] = X["PavedDrive"].map(
                {"N": 0, "P": 1, "Y": 2}
            )

        # ── STEP 7: CREATE NEW FEATURES ───────────────────
        # combining existing columns into more meaningful features
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

        X["TotalBaths"] = (
            X["FullBath"] + X["BsmtFullBath"] +
            X["HalfBath"] * 0.5 + X["BsmtHalfBath"] * 0.5
        )

        porch_cols = [c for c in ["OpenPorchSF", "EnclosedPorch",
                                   "3SsnPorch", "ScreenPorch"] if c in X.columns]
        X["TotalPorchSF"] = X[porch_cols].sum(axis=1)

        X["HouseAge"] = X["YrSold"].astype(int) - X["YearBuilt"]
        X["RemodAge"] = X["YrSold"].astype(int) - X["YearRemodAdd"]
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        if "PoolArea" in X.columns:
            X["HasPool"] = (X["PoolArea"] > 0).astype(int)

        # ── STEP 8: ONE-HOT ENCODING ──────────────────────
        # build all new columns first, then concat onc
        # avoids pandas fragmentation warning
        ohe_frames = []
        cols_to_drop_ohe = []

        for col in self.nominal_cols:
            if col in X.columns:
                for cat in self.ohe_categories_[col]:
                    ohe_frames.append(
                        (X[col] == cat).astype(int).rename(f"{col}_{cat}")
                    )
                cols_to_drop_ohe.append(col)

        X = X.drop(columns=cols_to_drop_ohe)

        if ohe_frames:
            X = pd.concat([X] + ohe_frames, axis=1)

        # ── STEP 9: FINAL SAFETY CHECK ────────────────────
        # drop any remaining string columns that slipped through
        remaining_str = X.select_dtypes(include=["object", "string"]).columns.tolist()
        if remaining_str:
            print(f"Warning: dropping remaining string cols: {remaining_str}")
            X = X.drop(columns=remaining_str)

        return X.copy()