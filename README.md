# 🧠 Diabetes Prediction

This machine learning project predicts the risk of diabetes using patient health data. It follows a full pipeline from data understanding and preprocessing to modeling and evaluation. Advanced techniques like clustering and ensemble learning were used to improve performance.

## 📂 Dataset
- **Source:** [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)

---

## 🚀 Project Workflow

### 1️⃣ Data Understanding
- Explored feature types and distributions
- Described numerical and categorical columns
- Checked for errors and inconsistencies

### 2️⃣ Data Cleaning
- Handled missing values, duplicates, and outliers

### 3️⃣ Feature Engineering & EDA
- Performed univariate and bivariate analysis
- Created new features based on domain knowledge
- Applied **KMeans clustering** and extracted new features by computing the distance between each data point and all cluster centers. These distance-based features capture patient similarity to different clusters instead of just assigning a single cluster label.

### 4️⃣ Preprocessing
- Encoding:
  - Ordinal: `OrdinalEncoder`, `LabelEncoder`
  - Nominal: `OneHotEncoder`, `BinaryEncoder`
- Handled class imbalance using:
  - **Over-Sampling** (SMOTE)
  - **Under-Sampling** (RandomUnderSampler)
- Applied different scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`

### 5️⃣ Modeling
- Trained multiple models: Logistic Regression, Random Forest, KNN, etc.
- For each model, tested both over-sampled and under-sampled versions to compare results
- Selected top 3 models based on validation performance
- Built a **Stacking Classifier** using these models for best results

### 6️⃣  Evaluation Focus: Recall Over Accuracy
In this medical problem, the key goal is to **identify as many diabetic patients as possible**.
- It’s more dangerous to predict someone is healthy when they actually have diabetes.
- Missing real cases means patients won’t get the treatment they need.
That's why I focused on optimizing **Recall**, not just Accuracy.
📈 Final model results:
- **Recall:** ~65%
- **Accuracy:** ~97%
- **Precision:** ~63%
- **fi score:** ~64% 
- **test score:** ~96%

### 7️⃣ Model Saving
- Saved final model using `joblib` for reuse or deployment

---

## 📊 Highlights

- ✅ Focused on **Recall** due to the medical context (high-risk if diabetic cases are missed)
- 🧠 Built full ML pipeline from data cleaning to modeling and deployment
- 📈 Achieved ~97% **Accuracy**, while optimizing **Recall**
- 🧩 Created a new feature using **KMeans clustering**
- ⚖️ Handled class imbalance using both **Over-Sampling (SMOTE)** and **Under-Sampling (Tomek Links)**
- 🧪 Evaluated every model with both sampling methods for fairness
- 🔀 Final model is a **Stacking Ensemble** of top 3 performing models
- 🌐 Built an interactive **Streamlit app** for real-time predictions

---

## 🧪 Streamlit App

To make the project interactive, I built a **Streamlit web app** where users can input their health data and get instant predictions.
