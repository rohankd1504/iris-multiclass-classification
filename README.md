# Iris Multiclass Classification Basics (Topic 31)
### GenAI Model Making Project — Group 16 | Class: LYCORE 2
**MIT School of Computing, MIT Art, Design and Technology University, Pune (2026–27)**

---

## 👥 Project Team Members
- **Rohan Kale** — Enrollment No: `ADT23SOCB0863`
- **Rohit Jagtap** — Enrollment No: `ADT23SOCB0867`
- **Rohan Dagade** — Enrollment No: `ADT23SOCB0864`
- **Rushikesh Shinde** — Enrollment No: `ADT23SOCB0896`

---

## 📁 Project Structure & Deliverables

```
REPORT AI/
├── Iris_Multiclass_Classification_Project_Report.pdf   <-- Complete Formatted Academic PDF Report (Ready to Print)
├── Iris_Multiclass_Classification_Project_Report.docx  <-- Complete Word Document Report
├── PROJECT_REPORT.md                                   <-- Full Markdown Report with Figures
├── app.py                                              <-- Interactive Streamlit Web Application
├── train_models.py                                     <-- Standalone Training & Evaluation Script
├── logistic_model.pkl                                  <-- Serialized Logistic Regression Model
├── knn_model.pkl                                       <-- Serialized kNN Model (100% Accuracy)
├── decision_tree_model.pkl                             <-- Serialized Decision Tree Model
├── mit_logo.png                                        <-- University Logo for Document Cover
├── figures/                                            <-- All 11 High-Res System & Evaluation Figures
│   ├── fig_5_1_block_diagram.png
│   ├── fig_6_1_flowchart.png
│   ├── fig_7_1_colab_setup.png
│   ├── fig_7_2_pairplot.png
│   ├── fig_7_3_model_training.png
│   ├── fig_7_4_accuracy_comparison.png
│   ├── fig_7_5_confusion_matrices.png
│   ├── fig_7_6_decision_boundaries.png
│   ├── fig_7_7_saved_models.png
│   ├── fig_7_8_streamlit_app.png
│   └── fig_7_9_prediction_output.png
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can:
1. Adjust the Sepal Length, Sepal Width, Petal Length, and Petal Width sliders.
2. Select your preferred model (kNN, Logistic Regression, or Decision Tree).
3. Click **Predict Flower Species** to get instant classification and probability scores.

### 2. Retrain Models & Re-generate Figures
```bash
python3 train_models.py
```

### 3. Re-generate Reports
- **Word Document (.docx):**
  ```bash
  python3 create_docx_report.py
  ```
- **PDF Report (.pdf):**
  ```bash
  python3 create_pdf_report.py
  ```

---

## 📊 Summary of Model Performance

| Algorithm | Scikit-learn Classifier | Testing Accuracy |
|---|---|---:|
| **k-Nearest Neighbors** | `KNeighborsClassifier(n_neighbors=5)` | **100.00%** |
| **Logistic Regression** | `LogisticRegression(max_iter=200)` | **96.67%** |
| **Decision Tree** | `DecisionTreeClassifier(random_state=42)` | **93.33%** |
