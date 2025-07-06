# 📊 OTT Platform Usage Among College Students – Survey Analysis

This project analyzes the usage patterns of Over-The-Top (OTT) platforms such as Netflix, Amazon Prime Video, and Disney+ Hotstar among college students. It includes data collection through a survey, data preprocessing, and the application of machine learning algorithms to classify and predict user behavior.

## 🧠 Objective

The goal is to:
- Understand viewing habits and preferences of college students.
- Analyze which OTT platforms are most used and how often.
- Explore the impact of OTT usage on students’ routines.
- Apply machine learning models to classify or predict OTT usage behavior.

## 📁 Dataset

- **Source**: Google Forms survey
- **File**: `OTT_Usage_Survey_Preprocessed.csv`
- **Rows**: 93 responses
- **Columns**: 12 features (e.g., Age, Gender, Preferred Platform, Time Spent, Subscription, etc.)
- **Target variable**: OTT user classification (e.g., High vs Low usage, Subscription type, etc.)

## 🧹 Data Cleaning

Steps performed:
- Removed duplicate entries.
- Filled null values using mean/mode.
- Standardized inconsistent text entries.
- Dropped irrelevant or non-useful columns.

## ⚙️ Preprocessing

- Categorical encoding using Label Encoding / One-Hot Encoding.
- Feature scaling using StandardScaler or MinMaxScaler.
- Split data into training and test sets (e.g., 80/20 ratio).

## 🤖 Algorithms Used

The following machine learning classification models were applied:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## 📈 Results

| Model                | Accuracy (%) |
|----------------------|--------------|
| Logistic Regression  | 76.34        |
| KNN                  | 72.58        |
| Decision Tree        | 78.49        |
| **Random Forest**    | **82.79**    |
| SVM                  | 75.27        |

✅ **Random Forest** performed the best with the highest accuracy.

## 📊 Visualizations

- Accuracy comparison bar graph
- Confusion matrix (for Random Forest)

> (Python code provided for generating graphs in Jupyter/VS Code.)

## 📄 Files Included

- `OTT_Usage_Survey_Preprocessed.csv` – Cleaned and prepared dataset.
- `OTT_Survey_Analysis_Report.docx` – Final written report.
- `notebook.ipynb` (optional) – Contains code for data analysis and modeling.
- `README.md` – Project overview.

## 🧾 Conclusion

This project demonstrates how survey data can be transformed into actionable insights using data science techniques. Machine learning models, especially Random Forest, were effective in classifying OTT user behavior. These findings can help content providers better understand youth trends and assist institutions in addressing digital media consumption patterns.

## 📌 Future Scope

- Expand survey to a larger demographic across different institutions.
- Explore clustering and segmentation techniques.
- Use deep learning models for enhanced prediction accuracy.

---

## 💡 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
