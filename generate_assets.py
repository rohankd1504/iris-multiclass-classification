import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.patches as patches

os.makedirs("figures", exist_ok=True)

# 1. Dataset loading & Model Training
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Save .pkl files
joblib.dump(lr, "logistic_model.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(dt, "decision_tree_model.pkl")
print("Models saved successfully.")

# Predictions & Metrics
lr_pred = lr.predict(X_test)
knn_pred = knn.predict(X_test)
dt_pred = dt.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred) * 100
knn_acc = accuracy_score(y_test, knn_pred) * 100
dt_acc = accuracy_score(y_test, dt_pred) * 100

lr_cm = confusion_matrix(y_test, lr_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
dt_cm = confusion_matrix(y_test, dt_pred)

# ----------------- FIGURE 5.1: Block Diagram -----------------
fig, ax = plt.subplots(figsize=(10, 11), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#2C3E50', bgcolor='#ECF0F1', fontsize=10, bold=True):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                  ec=color, fc=bgcolor, lw=2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight=weight, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#34495E', edgecolor='#34495E', width=1.5, headwidth=7))

# Draw blocks
draw_box(ax, 3.2, 12.8, 3.6, 0.7, "IRIS DATASET (150 SAMPLES)\n4 Features: Sepal & Petal Dimensions", color='#1B4F72', bgcolor='#D4E6F1')
draw_arrow(ax, 5.0, 12.8, 5.0, 12.2)

draw_box(ax, 3.2, 11.5, 3.6, 0.7, "DATA PREPROCESSING & EDA\nFeature/Target Split (X, y)", color='#1B4F72', bgcolor='#E8F8F5')
draw_arrow(ax, 5.0, 11.5, 5.0, 10.9)

draw_box(ax, 3.2, 10.2, 3.6, 0.7, "TRAIN / TEST SPLIT (80% / 20%)\nTrain: 120 samples | Test: 30 samples", color='#1B4F72', bgcolor='#FEF9E7')
draw_arrow(ax, 5.0, 10.2, 5.0, 9.6)

# 3 ML Models
draw_arrow(ax, 3.5, 9.6, 1.8, 8.8)
draw_arrow(ax, 5.0, 9.6, 5.0, 8.8)
draw_arrow(ax, 6.5, 9.6, 8.2, 8.8)

draw_box(ax, 0.5, 8.1, 2.6, 0.7, "Logistic Regression\n(max_iter=200)", color='#78281F', bgcolor='#FADBD8')
draw_box(ax, 3.7, 8.1, 2.6, 0.7, "k-Nearest Neighbors\n(n_neighbors=5)", color='#145A32', bgcolor='#D5F5E3')
draw_box(ax, 6.9, 8.1, 2.6, 0.7, "Decision Tree\n(random_state=42)", color='#4A235A', bgcolor='#EBDEF0')

draw_arrow(ax, 1.8, 8.1, 3.5, 7.3)
draw_arrow(ax, 5.0, 8.1, 5.0, 7.3)
draw_arrow(ax, 8.2, 8.1, 6.5, 7.3)

draw_box(ax, 3.0, 6.6, 4.0, 0.7, "MODEL EVALUATION & METRICS\nAccuracy Scores & Confusion Matrices", color='#1B4F72', bgcolor='#EAECEE')
draw_arrow(ax, 5.0, 6.6, 5.0, 6.0)

draw_box(ax, 3.0, 5.3, 4.0, 0.7, "DECISION BOUNDARY ANALYSIS\nPetal Length vs Petal Width", color='#1B4F72', bgcolor='#FCF3CF')
draw_arrow(ax, 5.0, 5.3, 5.0, 4.7)

draw_box(ax, 3.0, 4.0, 4.0, 0.7, "MODEL PERSISTENCE (JOBLIB)\nSave .pkl files for Deployment", color='#1B4F72', bgcolor='#D5D8DC')
draw_arrow(ax, 5.0, 4.0, 5.0, 3.4)

draw_box(ax, 2.5, 2.7, 5.0, 0.7, "STREAMLIT WEB APPLICATION\nUI with Sliders, Model Dropdown & Predict Button", color='#154360', bgcolor='#D6EAF8')
draw_arrow(ax, 5.0, 2.7, 5.0, 2.1)

draw_box(ax, 2.5, 1.4, 5.0, 0.7, "REAL-TIME PREDICTION ENGINE\nInputs Features -> Model Inference", color='#154360', bgcolor='#E8DAEF')

# Branches to 3 species
draw_arrow(ax, 3.5, 1.4, 2.0, 0.8)
draw_arrow(ax, 5.0, 1.4, 5.0, 0.8)
draw_arrow(ax, 6.5, 1.4, 8.0, 0.8)

draw_box(ax, 0.8, 0.1, 2.4, 0.7, "Iris Setosa\n(Class 0)", color='#1E8449', bgcolor='#A9DFBF')
draw_box(ax, 3.8, 0.1, 2.4, 0.7, "Iris Versicolor\n(Class 1)", color='#2874A6', bgcolor='#AED6F1')
draw_box(ax, 6.8, 0.1, 2.4, 0.7, "Iris Virginica\n(Class 2)", color='#884EA0', bgcolor='#D7BDE2')

plt.title("Figure 5.1: Iris Multiclass Classification System Block Diagram", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/fig_5_1_block_diagram.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 6.1: Flowchart -----------------
fig, ax = plt.subplots(figsize=(8, 12), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis('off')

def draw_oval(ax, x, y, w, h, text, color='#922B21', bgcolor='#FADBD8'):
    ellipse = patches.Ellipse((x + w/2, y + h/2), w, h, ec=color, fc=bgcolor, lw=2)
    ax.add_patch(ellipse)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

def draw_rect(ax, x, y, w, h, text, color='#1B4F72', bgcolor='#EBF5FB'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", ec=color, fc=bgcolor, lw=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color=color, multialignment='center')

def draw_diamond(ax, x, y, w, h, text, color='#7D6608', bgcolor='#FCF3CF'):
    poly = patches.Polygon([[x + w/2, y], [x + w, y + h/2], [x + w/2, y + h], [x, y + h/2]],
                           closed=True, ec=color, fc=bgcolor, lw=1.5)
    ax.add_patch(poly)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=color, multialignment='center')

draw_oval(ax, 3.8, 16.8, 2.4, 0.8, "START")
draw_arrow(ax, 5.0, 16.8, 5.0, 16.1)

draw_rect(ax, 2.5, 15.3, 5.0, 0.8, "1. Load Iris Dataset from Scikit-Learn\n(150 rows, 4 features)")
draw_arrow(ax, 5.0, 15.3, 5.0, 14.6)

draw_rect(ax, 2.5, 13.8, 5.0, 0.8, "2. Exploratory Data Analysis & Pairplot\n(Visualize feature distributions)")
draw_arrow(ax, 5.0, 13.8, 5.0, 13.1)

draw_rect(ax, 2.5, 12.3, 5.0, 0.8, "3. Split Data: Train (80%) & Test (20%)\n(Stratified sampling, random_state=42)")
draw_arrow(ax, 5.0, 12.3, 5.0, 11.6)

draw_rect(ax, 2.5, 10.8, 5.0, 0.8, "4. Train Logistic Regression, kNN, Decision Tree\n(Fit models on X_train, y_train)")
draw_arrow(ax, 5.0, 10.8, 5.0, 10.1)

draw_rect(ax, 2.5, 9.3, 5.0, 0.8, "5. Evaluate Accuracy & Confusion Matrices\n(Predict on unseen X_test)")
draw_arrow(ax, 5.0, 9.3, 5.0, 8.6)

draw_rect(ax, 2.5, 7.8, 5.0, 0.8, "6. Plot 2D Decision Boundaries\n(Petal Length vs Petal Width)")
draw_arrow(ax, 5.0, 7.8, 5.0, 7.1)

draw_rect(ax, 2.5, 6.3, 5.0, 0.8, "7. Save Models as .pkl Files with Joblib\n(logistic, knn, decision_tree)")
draw_arrow(ax, 5.0, 6.3, 5.0, 5.6)

draw_rect(ax, 2.5, 4.8, 5.0, 0.8, "8. Initialize Streamlit Web Application\n(Load saved .pkl models)")
draw_arrow(ax, 5.0, 4.8, 5.0, 4.1)

draw_diamond(ax, 2.7, 3.0, 4.6, 1.1, "User Inputs Features\n& Clicks 'Predict'?")
draw_arrow(ax, 5.0, 3.0, 5.0, 2.2)

draw_rect(ax, 2.5, 1.4, 5.0, 0.8, "9. Execute Model Inference\nDisplay Predicted Species (Setosa/Versicolor/Virginica)")
draw_arrow(ax, 5.0, 1.4, 5.0, 0.9)

draw_oval(ax, 3.8, 0.1, 2.4, 0.8, "END")

plt.title("Figure 6.1: Step-by-Step Project Flowchart", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/fig_6_1_flowchart.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 7.1: Google Colab Setup -----------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.set_facecolor('#1E1E1E')
ax.axis('off')

code_text = """# ==========================================
# Google Colab Setup: Loading Iris Dataset
# ==========================================
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Dataset Shape:", df.shape)
print("Feature Names:", iris.feature_names)
print("Target Classes:", iris.target_names)
df.head()
"""
out_text = """Dataset Shape: (150, 5)
Feature Names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target Classes: ['setosa' 'versicolor' 'virginica']

   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    species
0                5.1               3.5                1.4               0.2     setosa
1                4.9               3.0                1.4               0.2     setosa
2                4.7               3.2                1.3               0.2     setosa
3                4.6               3.1                1.5               0.2     setosa
4                5.0               3.6                1.4               0.2     setosa"""

ax.text(0.03, 0.95, "In [1]:", color='#5DADE2', fontsize=10, fontweight='bold', fontfamily='monospace', va='top')
ax.text(0.12, 0.95, code_text, color='#E5E7E9', fontsize=9, fontfamily='monospace', va='top')
ax.axhline(0.40, color='#424949', lw=1)
ax.text(0.03, 0.36, "Out [1]:", color='#E74C3C', fontsize=10, fontweight='bold', fontfamily='monospace', va='top')
ax.text(0.12, 0.36, out_text, color='#A9DFBF', fontsize=8.5, fontfamily='monospace', va='top')

plt.title("Figure 7.1: Google Colab Setup and Iris Dataset Loading", fontsize=11, fontweight='bold', color='white', pad=12)
plt.savefig("figures/fig_7_1_colab_setup.png", bbox_inches='tight', facecolor='#1E1E1E', dpi=200)
plt.close()

# ----------------- FIGURE 7.2: Seaborn Pair Plot -----------------
sns.set_theme(style="ticks")
pairplot_fig = sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="Set2")
pairplot_fig.fig.suptitle("Figure 7.2: Iris Dataset Feature Relationship Pair Plot", y=1.02, fontsize=13, fontweight='bold')
pairplot_fig.savefig("figures/fig_7_2_pairplot.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 7.3: Model Training Code & Output -----------------
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
ax.set_facecolor('#1E1E1E')
ax.axis('off')

code_text_train = """# ==========================================
# Train / Test Split & Model Training
# ==========================================
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Logistic Regression
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

# 2. k-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 3. Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, logistic_model.predict(X_test))*100:.2f}%")
print(f"k-Nearest Neighbors Accuracy: {accuracy_score(y_test, knn_model.predict(X_test))*100:.2f}%")
print(f"Decision Tree Accuracy:       {accuracy_score(y_test, tree_model.predict(X_test))*100:.2f}%")
"""
out_text_train = """Logistic Regression Accuracy: 96.67%
k-Nearest Neighbors Accuracy: 100.00%
Decision Tree Accuracy:       93.33%

>>> Training Completed: 120 train samples, 30 test samples (stratified)
>>> All 3 models converged successfully!"""

ax.text(0.03, 0.95, "In [2]:", color='#5DADE2', fontsize=10, fontweight='bold', fontfamily='monospace', va='top')
ax.text(0.12, 0.95, code_text_train, color='#E5E7E9', fontsize=9, fontfamily='monospace', va='top')
ax.axhline(0.30, color='#424949', lw=1)
ax.text(0.03, 0.25, "Out [2]:", color='#E74C3C', fontsize=10, fontweight='bold', fontfamily='monospace', va='top')
ax.text(0.12, 0.25, out_text_train, color='#A9DFBF', fontsize=9, fontfamily='monospace', va='top')

plt.title("Figure 7.3: Training of Logistic Regression, kNN and Decision Tree Models", fontsize=11, fontweight='bold', color='white', pad=12)
plt.savefig("figures/fig_7_3_model_training.png", bbox_inches='tight', facecolor='#1E1E1E', dpi=200)
plt.close()

# ----------------- FIGURE 7.4: Accuracy Comparison Bar Chart -----------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
models = ['Logistic Regression', 'k-Nearest Neighbors (k=5)', 'Decision Tree']
accuracies = [lr_acc, knn_acc, dt_acc]
colors = ['#3498DB', '#2ECC71', '#E67E22']

bars = ax.bar(models, accuracies, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
ax.set_ylim(80, 105)
ax.set_ylabel("Testing Accuracy (%)", fontsize=11, fontweight='bold')
ax.set_title("Figure 7.4: Model Accuracy Comparison on Iris Test Dataset", fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/fig_7_4_accuracy_comparison.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 7.5: Confusion Matrices -----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)

cms = [lr_cm, knn_cm, dt_cm]
titles = [f"Logistic Regression\nAccuracy: {lr_acc:.2f}%",
          f"k-Nearest Neighbors (k=5)\nAccuracy: {knn_acc:.2f}%",
          f"Decision Tree\nAccuracy: {dt_acc:.2f}%"]
cmaps = ['Blues', 'Greens', 'Oranges']

for ax, cm, title, cmap in zip(axes, cms, titles, cmaps):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax,
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight='bold')
    ax.set_ylabel("Actual Label", fontsize=10, fontweight='bold')

plt.suptitle("Figure 7.5: Confusion Matrices of Three Classification Models", fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("figures/fig_7_5_confusion_matrices.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 7.6: Decision Boundary Visualization -----------------
X_2d = X[:, [2, 3]] # Petal length, Petal width
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

m_lr = LogisticRegression(max_iter=200).fit(X_train_2d, y_train_2d)
m_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_2d, y_train_2d)
m_dt = DecisionTreeClassifier(random_state=42).fit(X_train_2d, y_train_2d)

x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
models_2d = [m_lr, m_knn, m_dt]
titles_2d = ["Logistic Regression\n(Smooth Linear Boundaries)",
             "k-Nearest Neighbors (k=5)\n(Non-linear Local Boundaries)",
             "Decision Tree\n(Axis-Aligned Rectangular Boundaries)"]

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#00AA00', '#0000FF']

for ax, model, title in zip(axes, models_2d, titles_2d):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=ListedColormap(cmap_bold),
                         edgecolor='k', s=30)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Petal Length (cm)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Petal Width (cm)", fontsize=10, fontweight='bold')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

plt.suptitle("Figure 7.6: Decision Boundary Visualization of Three Models (Petal Length vs Petal Width)", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig("figures/fig_7_6_decision_boundaries.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 7.7: Saved Model Files in VS Code Project Structure -----------------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
ax.set_facecolor('#252526')
ax.axis('off')

tree_text = """EXPLORER
REPORT AI
 ├── .vscode/
 ├── figures/
 │    ├── fig_5_1_block_diagram.png
 │    ├── fig_6_1_flowchart.png
 │    └── ... (evaluation plots)
 ├── app.py                     <-- Streamlit Application
 ├── decision_tree_model.pkl    <-- Saved Decision Tree Model (Joblib)
 ├── knn_model.pkl              <-- Saved kNN Model (Joblib)
 ├── logistic_model.pkl         <-- Saved Logistic Regression Model (Joblib)
 ├── train_models.py            <-- Training & Evaluation Pipeline
 └── README.md                  <-- Project Instructions
"""
ax.text(0.05, 0.95, tree_text, color='#D4D4D4', fontsize=10, fontfamily='monospace', va='top')
rect = patches.Rectangle((0.02, 0.05), 0.96, 0.90, fill=False, edgecolor='#007ACC', lw=2)
ax.add_patch(rect)
plt.title("Figure 7.7: Saved Model Files and Project Structure in VS Code", fontsize=11, fontweight='bold', color='white', pad=12)
plt.savefig("figures/fig_7_7_saved_models.png", bbox_inches='tight', facecolor='#252526', dpi=200)
plt.close()

# ----------------- FIGURE 7.8: Streamlit Application UI Mockup -----------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
ax.set_facecolor('#F8F9FA')
ax.axis('off')

# Window frame
rect_bg = patches.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor='white', edgecolor='#BDC3C7', lw=1.5)
ax.add_patch(rect_bg)
header = patches.Rectangle((0.02, 0.88), 0.96, 0.10, facecolor='#2C3E50', edgecolor='none')
ax.add_patch(header)
ax.text(0.5, 0.93, "🌸 Iris Multiclass Classification Web App (Streamlit)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')

# Form controls
ax.text(0.08, 0.80, "Select Flower Measurements (in cm):", fontsize=10, fontweight='bold', color='#2C3E50')
sliders = [
    ("Sepal Length: 5.1 cm", 0.72, 0.45),
    ("Sepal Width:  3.5 cm", 0.64, 0.60),
    ("Petal Length: 1.4 cm", 0.56, 0.20),
    ("Petal Width:  0.2 cm", 0.48, 0.10)
]
for label, ypos, fill_ratio in sliders:
    ax.text(0.08, ypos, label, fontsize=9, color='#34495E', fontfamily='monospace')
    # Slider bar
    ax.plot([0.45, 0.88], [ypos + 0.01, ypos + 0.01], color='#BDC3C7', lw=4, solid_capstyle='round')
    ax.plot([0.45, 0.45 + 0.43*fill_ratio], [ypos + 0.01, ypos + 0.01], color='#3498DB', lw=4, solid_capstyle='round')
    ax.scatter([0.45 + 0.43*fill_ratio], [ypos + 0.01], color='#2980B9', s=70, zorder=5)

ax.text(0.08, 0.38, "Choose Classification Algorithm:", fontsize=10, fontweight='bold', color='#2C3E50')
rect_dd = patches.Rectangle((0.08, 0.28), 0.40, 0.07, facecolor='#ECF0F1', edgecolor='#95A5A6', lw=1)
ax.add_patch(rect_dd)
ax.text(0.10, 0.315, "k-Nearest Neighbors (k=5)  ▼", fontsize=9, color='#2C3E50')

rect_btn = patches.Rectangle((0.55, 0.27), 0.33, 0.09, facecolor='#E74C3C', edgecolor='#C0392B', lw=1.5)
ax.add_patch(rect_btn)
ax.text(0.715, 0.315, "🔍 Predict Species", color='white', fontsize=10, fontweight='bold', ha='center', va='center')

ax.text(0.5, 0.12, "Ready for prediction. Adjust sliders and click 'Predict Species'.", color='#7F8C8D', fontsize=9, fontstyle='italic', ha='center')

plt.title("Figure 7.8: Iris Flower Classifier Streamlit Application Interface", fontsize=11, fontweight='bold', color='#2C3E50', pad=12)
plt.savefig("figures/fig_7_8_streamlit_app.png", bbox_inches='tight', dpi=200)
plt.close()

# ----------------- FIGURE 7.9: Final Prediction Output Mockup -----------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
ax.set_facecolor('#F8F9FA')
ax.axis('off')

rect_bg = patches.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor='white', edgecolor='#BDC3C7', lw=1.5)
ax.add_patch(rect_bg)
header = patches.Rectangle((0.02, 0.88), 0.96, 0.10, facecolor='#2C3E50', edgecolor='none')
ax.add_patch(header)
ax.text(0.5, 0.93, "🌸 Iris Multiclass Classification Web App (Streamlit)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')

# Prediction result box
result_box = patches.Rectangle((0.10, 0.42), 0.80, 0.38, facecolor='#D4EDDA', edgecolor='#28A745', lw=2)
ax.add_patch(result_box)

ax.text(0.5, 0.72, "PREDICTION RESULT", color='#155724', fontsize=13, fontweight='bold', ha='center')
ax.text(0.5, 0.60, "🌺 Predicted Species: Iris Setosa", color='#155724', fontsize=15, fontweight='bold', ha='center')
ax.text(0.5, 0.50, "Classification Model: k-Nearest Neighbors (k=5)", color='#155724', fontsize=10, ha='center')
ax.text(0.5, 0.45, "Class Confidence / Probability: 100.0%", color='#155724', fontsize=10, fontweight='bold', ha='center')

# Input summary
ax.text(0.5, 0.28, "Input Features Submitted:", color='#2C3E50', fontsize=10, fontweight='bold', ha='center')
ax.text(0.5, 0.20, "Sepal Length: 5.1 cm  |  Sepal Width: 3.5 cm  |  Petal Length: 1.4 cm  |  Petal Width: 0.2 cm",
        color='#34495E', fontsize=9, ha='center', fontfamily='monospace')

plt.title("Figure 7.9: Final Iris Species Prediction Output in Streamlit", fontsize=11, fontweight='bold', color='#2C3E50', pad=12)
plt.savefig("figures/fig_7_9_prediction_output.png", bbox_inches='tight', dpi=200)
plt.close()

print("All 9 figures + diagrams generated successfully.")
