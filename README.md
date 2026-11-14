# Insurance Claims Fraud Detection

A machine learning project that scores auto insurance claims for fraud risk, with a live web demo hosted on Vercel.

## Business problem
Fraudulent claims cost insurers money, and investigators have limited time. This model gives each claim a fraud probability so the riskiest claims are reviewed first. The decision threshold favours recall (F2 score), because missing a fraudulent claim costs more than reviewing an extra one.

## Project structure
```
vercel-fraud-detection/
├── public/                  <- what Vercel hosts (static site)
│   ├── index.html           <- web app, scores claims in the browser
│   └── model.json           <- exported model (DEMO until you retrain)
├── training/                <- run locally, never deployed
│   ├── fraud_detection.py   <- full analysis: 3 models, CV, metrics, plots
│   ├── export_for_vercel.py <- exports the model to public/model.json
│   ├── app.py               <- optional local Streamlit demo
│   ├── requirements.txt
│   └── data/                <- put insurance_claims.csv here
├── vercel.json
└── README.md
```

## Why a static site and not Streamlit?
Vercel runs short-lived serverless functions, so it cannot host Streamlit (which needs a long-running server), and scikit-learn is too heavy for serverless bundles. Instead, the Logistic Regression coefficients are exported to JSON and the scoring formula (scale, one-hot encode, weighted sum, sigmoid) runs in the browser. It gives the same predictions as scikit-learn (checked to about 1e-13) and loads instantly.

## Step 1: Train on the real data
1. Download `insurance_claims.csv` from Kaggle (search "Auto Insurance Claims Data") into `training/data/`.
2. From the `training/` folder:
   ```
   pip install -r requirements.txt
   python fraud_detection.py --data data/insurance_claims.csv
   python export_for_vercel.py --data data/insurance_claims.csv
   ```
   - The first command compares the models, saves charts to `training/outputs/` and prints the metrics. Use these for your README results and interview.
   - The second overwrites `public/model.json` with your real model and removes the demo banner.

## Step 2: Deploy to Vercel
**Option A: GitHub (recommended)**
1. Push this folder to a GitHub repository.
2. Go to vercel.com, click **Add New > Project**, and import the repository.
3. Leave the framework as **Other** and the build command empty. `vercel.json` already sets the output directory to `public`.
4. Click **Deploy**. You get a link like `your-project.vercel.app`.

**Option B: CLI**
```
npm i -g vercel
vercel --prod
```

## Test locally before deploying
```
cd public
python -m http.server 8000
```
Open http://localhost:8000.

## Results (fill in after training)
| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| Logistic Regression | | |
| Random Forest | | |
| Gradient Boosting | | |

Add `training/outputs/evaluation.png` and `feature_importance.png`, and 3 or 4 key insights.

## Notes for interviews
- Why PR-AUC and recall instead of accuracy: fraud is an imbalanced class.
- Why a Pipeline: preprocessing is fit only on training folds, so there is no data leakage.
- Why Logistic Regression for the web version: it is interpretable (important in regulated insurance) and can be exported as plain coefficients.
- Limitations: one public dataset, and the model is for prioritising reviews, not for making final decisions about claimants.
- Next steps: SHAP explanations, hyperparameter tuning, FastAPI backend, monitoring for drift.
