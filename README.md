# Heart Disease Prediction (PRCP-1016)

## Project Overview

Heart disease remains one of the leading causes of death globally. Early detection and accurate risk prediction can significantly improve patient outcomes and reduce mortality.

This project develops a machine learning classification system to predict the presence of heart disease based on clinical and medical attributes. The goal is to support healthcare professionals with data-driven insights for early diagnosis and intervention.

**Target Variable:**  
`heart_disease_present`  
- `0` → No heart disease  
- `1` → Heart disease present  

**Primary Objectives:**
- Perform thorough Exploratory Data Analysis (EDA)
- Build and compare multiple classification models
- Select the best-performing model based on clinically relevant metrics
- Deliver actionable healthcare insights

---

## Dataset

- **Source**: PRCP-1016 Heart Disease Prediction dataset
- **Files**:
  - `values.csv` – Clinical features
  - `labels.csv` – Target labels
- **Merge Key**: `patient_id`
- **Final Shape**: 180 patients × 15 columns (after merge)

### Feature Description

| Feature | Description |
|---------|-------------|
| `slope_of_peak_exercise_st_segment` | Slope of the peak exercise ST segment (1–3) |
| `thal` | Thalassemia status (`normal`, `fixed_defect`, `reversible_defect`) |
| `resting_blood_pressure` | Resting blood pressure (mm Hg) |
| `chest_pain_type` | Type of chest pain (1–4) |
| `num_major_vessels` | Number of major vessels colored by fluoroscopy (0–3) |
| `fasting_blood_sugar_gt_120_mg_per_dl` | Fasting blood sugar > 120 mg/dl (0/1) |
| `resting_ekg_results` | Resting electrocardiographic results (0–2) |
| `serum_cholesterol_mg_per_dl` | Serum cholesterol (mg/dl) |
| `oldpeak_eq_st_depression` | ST depression induced by exercise relative to rest |
| `sex` | Sex (0 = female, 1 = male) |
| `age` | Age in years |
| `max_heart_rate_achieved` | Maximum heart rate achieved |
| `exercise_induced_angina` | Exercise-induced angina (0/1) |
| `heart_disease_present` | Target (0/1) |

---

## Project Pipeline

1. **Imports & Setup** – Core libraries, visualization, preprocessing, models, and metrics
2. **Data Loading & Merging** – Download, unzip, and merge `values.csv` + `labels.csv`
3. **Exploratory Data Analysis (EDA)** – Distributions, correlations, categorical analysis, and target relationships
4. **Data Preprocessing** – Encoding categorical variables, handling mixed data types, feature scaling
5. **Train-Test Split** – Stratified split to preserve class balance
6. **Model Building** – Multiple classification algorithms
7. **Model Evaluation** – Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
8. **Hyperparameter Tuning** – GridSearchCV on key models
9. **Final Model Selection** – Based on clinical relevance (especially Recall and ROC-AUC)
10. **Model Persistence** – Saved as `final_heart_model.pkl`
11. **Healthcare Insights & Recommendations**

---

## Models Evaluated

| Model                        | Type                     | Notes                                      |
|-----------------------------|--------------------------|--------------------------------------------|
| Logistic Regression         | Linear                   | Interpretable baseline                     |
| Decision Tree               | Tree-based               | Easy to interpret                          |
| Random Forest               | Ensemble (Bagging)       | Robust performance                         |
| Gradient Boosting           | Ensemble (Boosting)      | Strong predictive power                    |
| Support Vector Machine (SVM)| Kernel-based             | Effective in high-dimensional spaces       |
| K-Nearest Neighbors (KNN)   | Instance-based           | Simple but sensitive to scaling            |
| **Gaussian Naive Bayes**    | Probabilistic            | **Selected as final model**                |
| XGBoost                     | Advanced Boosting        | High performance after tuning              |

### Hyperparameter Tuning (GridSearchCV – Scoring: ROC-AUC)

- **Logistic Regression**: `C`, `solver`
- **Decision Tree**: `max_depth`, `min_samples_split`
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
- **Gradient Boosting**: `n_estimators`, `learning_rate`, `max_depth`
- **SVM**: `C`, `kernel`
- **KNN**: `n_neighbors`, `weights`
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`
- **Naive Bayes**: `var_smoothing`

---

## Final Model Performance

**Selected Model: Gaussian Naive Bayes**

| Metric       | Score     |
|--------------|-----------|
| **Accuracy** | 0.8333    |
| **Precision**| 0.8571    |
| **Recall**   | 0.7500    |
| **F1-Score** | 0.8000    |
| **ROC-AUC**  | **0.9111**|

**Confusion Matrix (Test Set):**
```
[[27  3]
 [ 6 18]]
```

### Why Naive Bayes Was Selected
- Highest ROC-AUC (0.9111) among evaluated models
- Strong Recall (0.75) — critical in healthcare to minimize false negatives
- Balanced overall performance
- Low model complexity and high stability
- Fast inference suitable for clinical decision support

> **Clinical Priority**: In heart disease prediction, **Recall** is often more important than pure accuracy because missing a positive case (false negative) can have severe consequences.

---

## Key Healthcare Insights

- Patients with higher **chest pain type** values are at significantly elevated risk.
- **Exercise-induced angina** is a strong clinical indicator of underlying heart disease.
- Combination of elevated **serum cholesterol** and high **resting blood pressure** increases disease probability.
- Features such as `oldpeak` (ST depression), `num_major_vessels`, and `thal` status provide valuable diagnostic signals.
- Early detection using these clinical markers can substantially reduce mortality risk.

---

## Challenges & Solutions

| Challenge                        | Solution Applied                              |
|----------------------------------|-----------------------------------------------|
| Mixed data types (numeric + categorical) | Label / One-hot encoding                     |
| Different feature scales         | StandardScaler for distance-based models      |
| Model selection uncertainty      | Systematic comparison of 8 algorithms         |
| Risk of overfitting              | Cross-validation + GridSearchCV               |
| Small dataset size (n=180)       | Emphasis on robust metrics (ROC-AUC, Recall)  |

---

## How to Reproduce

### Requirements
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

### Steps
1. Open the Jupyter notebook `PRCP-1016-HeartDieseasePred.ipynb`
2. Run the data download and unzip cells (or place the Data folder manually)
3. Execute cells sequentially through preprocessing, modeling, tuning, and evaluation
4. The final model is saved as:
   ```python
   pickle.dump(final_model, open("final_heart_model.pkl", "wb"))
   ```

### Loading the Saved Model
```python
import pickle

model = pickle.load(open("final_heart_model.pkl", "rb"))
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

## Project Structure

```
├── PRCP-1016-HeartDieseasePred.ipynb   # Main analysis notebook
├── final_heart_model.pkl               # Trained Naive Bayes model
├── Data/
│   ├── values.csv                      # Clinical features
│   ├── labels.csv                      # Target labels
│   └── description.docx                # Feature documentation
└── README.md              # This file
```

---

## Conclusion

This project successfully demonstrates an end-to-end machine learning workflow for heart disease prediction. Through careful data preparation, multi-model comparison, and clinically oriented evaluation, **Gaussian Naive Bayes** emerged as the best model with a strong ROC-AUC of **0.911** and solid recall.

The resulting system can serve as a decision-support tool to assist healthcare professionals in identifying high-risk patients earlier, enabling timely intervention and potentially saving lives.

---

**Project Code**: PRCP-1016  
**Domain**: Healthcare / Cardiology  
**Task Type**: Binary Classification  
**Final Model**: Gaussian Naive Bayes  
**Key Metric**: ROC-AUC = 0.9111
