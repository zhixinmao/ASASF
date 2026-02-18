{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ef2da7db",
   "metadata": {},
   "source": [
    "# Prediction of HDL on NHANES data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "52878da6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import KFold, RandomizedSearchCV\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.ensemble import ExtraTreesRegressor\n",
    "from sklearn.ensemble import HistGradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.base import clone\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "RANDOM_SEED = 7\n",
    "np.random.seed(RANDOM_SEED)\n",
    "\n",
    "train = pd.read_csv(\"train_dat.csv\")\n",
    "train = train.iloc[:,1:]\n",
    "test = pd.read_csv(\"test_dat.csv\")\n",
    "test = test.iloc[:,1:]\n",
    "labels = pd.read_csv(\"train_variable_labels.csv\")\n",
    "OUTCOME = \"LBDHDD_outcome\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "25f30653",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.hist(train[OUTCOME].astype(float), bins=30)\n",
    "plt.xlabel(OUTCOME)\n",
    "plt.ylabel(\"Count\")\n",
    "plt.title(\"Outcome distribution: LBDHDD_outcome\")\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "610ab5e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# data preprocessing\n",
    "def infer_categorical_columns(df: pd.DataFrame, outcome_col: str):\n",
    "    \"\"\"\n",
    "    Heuristic: NHANES-style coded vars tend to be low-cardinality ints.\n",
    "    We'll treat:\n",
    "      - integer columns with <=20 unique values as categorical\n",
    "      - plus some common prefixes with moderate unique counts\n",
    "    \"\"\"\n",
    "    cat_cols = []\n",
    "    for c in df.columns:\n",
    "        if c == outcome_col:\n",
    "            continue\n",
    "        s = df[c]\n",
    "        if pd.api.types.is_integer_dtype(s) and s.nunique() <= 20:\n",
    "            cat_cols.append(c)\n",
    "            continue\n",
    "        name = c.upper()\n",
    "        if name.startswith((\"RIAG\", \"RID\", \"DMD\", \"DRQ\", \"ALQ\", \"SMQ\", \"PAQ\")) and s.nunique() <= 60:\n",
    "            cat_cols.append(c)\n",
    "            continue\n",
    "        lab = str(label_map.get(c, \"\")).lower()\n",
    "        kw = [\"source\",\"status\",\"day\",\"eaten\",\"used\",\"help\",\"compare\",\"frequency\",\"how often\",\"marital\",\"race\",\"gender\",\"sex\",\"education\"]\n",
    "        if any(k in lab for k in kw) and s.nunique() <= 60:\n",
    "            cat_cols.append(c)\n",
    "    cat_cols = sorted(set(cat_cols))\n",
    "    num_cols = [c for c in df.columns if c != outcome_col and pd.api.types.is_numeric_dtype(df[c])]\n",
    "    cont_cols = sorted([c for c in num_cols if c not in cat_cols])\n",
    "    return cat_cols, cont_cols\n",
    "\n",
    "# NHANES-ish special codes often represent \"Refused/Don't know/Missing/Not applicable\"\n",
    "CAT_MISSING = {77, 88, 99}\n",
    "BIG_MISSING = {777, 888, 999, 7777, 8888, 9999, 77777, 88888, 99999}\n",
    "\n",
    "def recode_special_missing(df: pd.DataFrame, cat_cols, cont_cols):\n",
    "    df = df.copy()\n",
    "    # categorical special codes -> NaN\n",
    "    for c in cat_cols:\n",
    "        if c in df.columns:\n",
    "            df[c] = df[c].where(~df[c].isin(CAT_MISSING), np.nan)\n",
    "\n",
    "    # continuous: large sentinel codes -> NaN\n",
    "    for c in cont_cols:\n",
    "        if c in df.columns:\n",
    "            df[c] = df[c].where(~df[c].isin(BIG_MISSING), np.nan)\n",
    "    return df\n",
    "\n",
    "# Label file in your upload has a typo in column name: \"variale\"\n",
    "label_map = dict(zip(labels[\"variale\"], labels[\"label\"]))\n",
    "\n",
    "# remove zero variance columns\n",
    "train1 = train.drop(columns = [\"DRABF\", \"DR1MRESP\", \"ALQ111\"], errors=\"ignore\")\n",
    "test1 = test.drop(columns = [\"DRABF\", \"DR1MRESP\", \"ALQ111\"], errors=\"ignore\")\n",
    "\n",
    "cat_cols, cont_cols = infer_categorical_columns(train1, OUTCOME)\n",
    "train1 = recode_special_missing(train1, cat_cols, cont_cols)\n",
    "test1  = recode_special_missing(test1, cat_cols, cont_cols)\n",
    "\n",
    "X = train1.drop(columns=[OUTCOME])\n",
    "y = train1[OUTCOME].astype(float).values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "18958c69",
   "metadata": {},
   "outputs": [],
   "source": [
    "# columnTransformer\n",
    "numeric_pipe = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "    (\"scaler\", StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_pipe = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
    "    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\", sparse_output=True))\n",
    "])\n",
    "\n",
    "preprocess = ColumnTransformer(\n",
    "    transformers=[\n",
    "        (\"num\", numeric_pipe, cont_cols),\n",
    "        (\"cat\", categorical_pipe, cat_cols),\n",
    "    ],\n",
    "    remainder=\"drop\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a812d230",
   "metadata": {},
   "outputs": [],
   "source": [
    "# build models\n",
    "ridge = Pipeline(steps=[\n",
    "    (\"prep\", preprocess),\n",
    "    (\"model\", Ridge(random_state=RANDOM_SEED))\n",
    "])\n",
    "\n",
    "hgb = Pipeline(steps=[\n",
    "    (\"prep\", preprocess),\n",
    "    (\"model\", HistGradientBoostingRegressor(random_state=RANDOM_SEED))\n",
    "])\n",
    "\n",
    "etr = Pipeline(steps=[\n",
    "    (\"prep\", preprocess),\n",
    "    (\"model\", ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1))\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "3cdb92da",
   "metadata": {},
   "outputs": [],
   "source": [
    "# cv utility\n",
    "def rmse(y_true, y_pred):\n",
    "    return np.sqrt(mean_squared_error(y_true, y_pred))\n",
    "\n",
    "def oof_predictions(model, X, y, cv):\n",
    "    oof = np.zeros(len(y), dtype=float)\n",
    "    for fold, (tr, va) in enumerate(cv.split(X,y), 1):\n",
    "        m = clone(model)\n",
    "        m.fit(X.iloc[tr], y[tr])\n",
    "        oof[va] = m.predict(X.iloc[va])\n",
    "    return(oof)\n",
    "\n",
    "cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ca5b230a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CV RMSE (best):\n",
      "  Ridge: 5.629005564949829\n",
      "  HGB:   4.812756512073859\n",
      "  ETR:   5.000539634519933\n"
     ]
    }
   ],
   "source": [
    "# tune parameters\n",
    "ridge_params = {\n",
    "    \"model__alpha\": np.logspace(-3, 3, 50)\n",
    "}\n",
    "\n",
    "hgb_params = {\n",
    "    \"model__learning_rate\": np.linspace(0.02, 0.2, 10),\n",
    "    \"model__max_depth\": [2, 3, 4, None],\n",
    "    \"model__max_leaf_nodes\": [15, 31, 63],\n",
    "    \"model__min_samples_leaf\": [10, 20, 30, 50],\n",
    "    \"model__l2_regularization\": np.logspace(-4, 1, 10),\n",
    "}\n",
    "\n",
    "etr_params = {\n",
    "    \"model__n_estimators\": [300, 600, 1000],\n",
    "    \"model__max_depth\": [None, 6, 10, 14],\n",
    "    \"model__min_samples_split\": [2, 5, 10],\n",
    "    \"model__min_samples_leaf\": [1, 2, 5],\n",
    "    \"model__max_features\": [\"sqrt\", 0.5, 0.8],\n",
    "}\n",
    "\n",
    "def tune(model, params, X, y, cv, n_iter=30):\n",
    "    search = RandomizedSearchCV(\n",
    "        model,\n",
    "        param_distributions=params,\n",
    "        n_iter=n_iter,\n",
    "        scoring=\"neg_root_mean_squared_error\",\n",
    "        cv=cv,\n",
    "        random_state=RANDOM_SEED,\n",
    "        n_jobs=-1,\n",
    "        verbose=0\n",
    "    )\n",
    "    search.fit(X, y)\n",
    "    return search.best_estimator_, -search.best_score_\n",
    "\n",
    "# Tune each model\n",
    "ridge_best, ridge_cv_rmse = tune(ridge, ridge_params, X, y, cv, n_iter=25)\n",
    "hgb_best,   hgb_cv_rmse   = tune(hgb,   hgb_params,   X, y, cv, n_iter=40)\n",
    "etr_best,   etr_cv_rmse   = tune(etr,   etr_params,   X, y, cv, n_iter=40)\n",
    "\n",
    "print(\"CV RMSE (best):\")\n",
    "print(\"  Ridge:\", ridge_cv_rmse)\n",
    "print(\"  HGB:  \", hgb_cv_rmse)\n",
    "print(\"  ETR:  \", etr_cv_rmse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "237dcda4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best ensemble (OOF) RMSE: 4.760772044632013 weights: (0.1, 0.72, 0.18000000000000005)\n"
     ]
    }
   ],
   "source": [
    "# build OOF predicions + choose ensemble weights\n",
    "oof_ridge = oof_predictions(ridge_best, X, y, cv)\n",
    "oof_hgb = oof_predictions(hgb_best, X, y, cv)\n",
    "oof_etr = oof_predictions(etr_best, X, y, cv)\n",
    "\n",
    "weights = []\n",
    "for a in np.linspace(0, 1, 11):\n",
    "    for b in np.linspace(0, 1-a, 11):\n",
    "        c = 1-a-b\n",
    "        pred = a*oof_ridge + b*oof_hgb + c*oof_etr\n",
    "        weights.append((rmse(y, pred), a, b, c))\n",
    "weights.sort()\n",
    "best_rmse, wa, wb, wc = weights[0]\n",
    "print(\"Best ensemble (OOF) RMSE:\", best_rmse, \"weights:\", (wa, wb, wc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c7c9a848",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wrote pred.csv with shape: (200, 1)\n"
     ]
    }
   ],
   "source": [
    "# fit the final models on full train, predict test\n",
    "ridge_best.fit(X, y)\n",
    "hgb_best.fit(X, y)\n",
    "etr_best.fit(X, y)\n",
    "\n",
    "x_test = test1.copy()\n",
    "pred_test = wa*ridge_best.predict(x_test) + wb*hgb_best.predict(x_test) + wc*etr_best.predict(x_test)\n",
    "\n",
    "# Submission format\n",
    "pred_df = pd.DataFrame({\"pred\": pred_test})\n",
    "pred_df.to_csv(\"pred.csv\", index=False)\n",
    "print(\"Wrote pred.csv with shape:\", pred_df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "479af63f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>49.665575</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>53.942283</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>59.623769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>48.889081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>43.015755</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>195</th>\n",
       "      <td>46.418255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>196</th>\n",
       "      <td>45.867496</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>197</th>\n",
       "      <td>49.642425</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>198</th>\n",
       "      <td>48.502068</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>199</th>\n",
       "      <td>63.251935</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>200 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          pred\n",
       "0    49.665575\n",
       "1    53.942283\n",
       "2    59.623769\n",
       "3    48.889081\n",
       "4    43.015755\n",
       "..         ...\n",
       "195  46.418255\n",
       "196  45.867496\n",
       "197  49.642425\n",
       "198  48.502068\n",
       "199  63.251935\n",
       "\n",
       "[200 rows x 1 columns]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a470dc31",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
