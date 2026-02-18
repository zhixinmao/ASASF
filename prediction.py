#!/usr/bin/env python
# coding: utf-8

# # Prediction of HDL on NHANES data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.impute import SimpleImputer

RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

train = pd.read_csv("train_dat.csv")
train = train.iloc[:,1:]
test = pd.read_csv("test_dat.csv")
test = test.iloc[:,1:]
labels = pd.read_csv("train_variable_labels.csv")
OUTCOME = "LBDHDD_outcome"


# In[2]:


plt.figure()
plt.hist(train[OUTCOME].astype(float), bins=30)
plt.xlabel(OUTCOME)
plt.ylabel("Count")
plt.title("Outcome distribution: LBDHDD_outcome")
plt.tight_layout()


# In[3]:


# data preprocessing
def infer_categorical_columns(df: pd.DataFrame, outcome_col: str):
    """
    Heuristic: NHANES-style coded vars tend to be low-cardinality ints.
    We'll treat:
      - integer columns with <=20 unique values as categorical
      - plus some common prefixes with moderate unique counts
    """
    cat_cols = []
    for c in df.columns:
        if c == outcome_col:
            continue
        s = df[c]
        if pd.api.types.is_integer_dtype(s) and s.nunique() <= 20:
            cat_cols.append(c)
            continue
        name = c.upper()
        if name.startswith(("RIAG", "RID", "DMD", "DRQ", "ALQ", "SMQ", "PAQ")) and s.nunique() <= 60:
            cat_cols.append(c)
            continue
        lab = str(label_map.get(c, "")).lower()
        kw = ["source","status","day","eaten","used","help","compare","frequency","how often","marital","race","gender","sex","education"]
        if any(k in lab for k in kw) and s.nunique() <= 60:
            cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in df.columns if c != outcome_col and pd.api.types.is_numeric_dtype(df[c])]
    cont_cols = sorted([c for c in num_cols if c not in cat_cols])
    return cat_cols, cont_cols

# NHANES-ish special codes often represent "Refused/Don't know/Missing/Not applicable"
CAT_MISSING = {77, 88, 99}
BIG_MISSING = {777, 888, 999, 7777, 8888, 9999, 77777, 88888, 99999}

def recode_special_missing(df: pd.DataFrame, cat_cols, cont_cols):
    df = df.copy()
    # categorical special codes -> NaN
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].where(~df[c].isin(CAT_MISSING), np.nan)

    # continuous: large sentinel codes -> NaN
    for c in cont_cols:
        if c in df.columns:
            df[c] = df[c].where(~df[c].isin(BIG_MISSING), np.nan)
    return df

# Label file in your upload has a typo in column name: "variale"
label_map = dict(zip(labels["variale"], labels["label"]))

# remove zero variance columns
train1 = train.drop(columns = ["DRABF", "DR1MRESP", "ALQ111"], errors="ignore")
test1 = test.drop(columns = ["DRABF", "DR1MRESP", "ALQ111"], errors="ignore")

cat_cols, cont_cols = infer_categorical_columns(train1, OUTCOME)
train1 = recode_special_missing(train1, cat_cols, cont_cols)
test1  = recode_special_missing(test1, cat_cols, cont_cols)

X = train1.drop(columns=[OUTCOME])
y = train1[OUTCOME].astype(float).values


# In[4]:


# columnTransformer
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, cont_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)


# In[5]:


# build models
ridge = Pipeline(steps=[
    ("prep", preprocess),
    ("model", Ridge(random_state=RANDOM_SEED))
])

hgb = Pipeline(steps=[
    ("prep", preprocess),
    ("model", HistGradientBoostingRegressor(random_state=RANDOM_SEED))
])

etr = Pipeline(steps=[
    ("prep", preprocess),
    ("model", ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1))
])


# In[6]:


# cv utility
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def oof_predictions(model, X, y, cv):
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(cv.split(X,y), 1):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    return(oof)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


# In[8]:


# tune parameters
ridge_params = {
    "model__alpha": np.logspace(-3, 3, 50)
}

hgb_params = {
    "model__learning_rate": np.linspace(0.02, 0.2, 10),
    "model__max_depth": [2, 3, 4, None],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 30, 50],
    "model__l2_regularization": np.logspace(-4, 1, 10),
}

etr_params = {
    "model__n_estimators": [300, 600, 1000],
    "model__max_depth": [None, 6, 10, 14],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", 0.5, 0.8],
}

def tune(model, params, X, y, cv, n_iter=30):
    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X, y)
    return search.best_estimator_, -search.best_score_

# Tune each model
ridge_best, ridge_cv_rmse = tune(ridge, ridge_params, X, y, cv, n_iter=25)
hgb_best,   hgb_cv_rmse   = tune(hgb,   hgb_params,   X, y, cv, n_iter=40)
etr_best,   etr_cv_rmse   = tune(etr,   etr_params,   X, y, cv, n_iter=40)

print("CV RMSE (best):")
print("  Ridge:", ridge_cv_rmse)
print("  HGB:  ", hgb_cv_rmse)
print("  ETR:  ", etr_cv_rmse)


# In[ ]:


# build OOF predicions + choose ensemble weights
oof_ridge = oof_predictions(ridge_best, X, y, cv)
oof_hgb = oof_predictions(hgb_best, X, y, cv)
oof_etr = oof_predictions(etr_best, X, y, cv)

weights = []
for a in np.linspace(0, 1, 11):
    for b in np.linspace(0, 1-a, 11):
        c = 1-a-b
        pred = a*oof_ridge + b*oof_hgb + c*oof_etr
        weights.append((rmse(y, pred), a, b, c))
weights.sort()
best_rmse, wa, wb, wc = weights[0]
print("Best ensemble (OOF) RMSE:", best_rmse, "weights:", (wa, wb, wc))


# In[21]:


# fit the final models on full train, predict test
ridge_best.fit(X, y)
hgb_best.fit(X, y)
etr_best.fit(X, y)

x_test = test1.copy()
pred_test = wa*ridge_best.predict(x_test) + wb*hgb_best.predict(x_test) + wc*etr_best.predict(x_test)

# Submission format
pred_df = pd.DataFrame({"pred": pred_test})
pred_df.to_csv("pred.csv", index=False)
print("Wrote pred.csv with shape:", pred_df.shape)


# In[22]:


pred_df


# In[ ]:




