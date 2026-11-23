import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Update fig 7.8 without emoji warnings
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
ax.set_facecolor('#F8F9FA')
ax.axis('off')

rect_bg = patches.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor='white', edgecolor='#BDC3C7', lw=1.5)
ax.add_patch(rect_bg)
header = patches.Rectangle((0.02, 0.88), 0.96, 0.10, facecolor='#2C3E50', edgecolor='none')
ax.add_patch(header)
ax.text(0.5, 0.93, "Iris Multiclass Classification Web App (Streamlit)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')

ax.text(0.08, 0.80, "Select Flower Measurements (in cm):", fontsize=10, fontweight='bold', color='#2C3E50')
sliders = [
    ("Sepal Length: 5.1 cm", 0.72, 0.45),
    ("Sepal Width:  3.5 cm", 0.64, 0.60),
    ("Petal Length: 1.4 cm", 0.56, 0.20),
    ("Petal Width:  0.2 cm", 0.48, 0.10)
]
for label, ypos, fill_ratio in sliders:
    ax.text(0.08, ypos, label, fontsize=9, color='#34495E', fontfamily='monospace')
    ax.plot([0.45, 0.88], [ypos + 0.01, ypos + 0.01], color='#BDC3C7', lw=4, solid_capstyle='round')
    ax.plot([0.45, 0.45 + 0.43*fill_ratio], [ypos + 0.01, ypos + 0.01], color='#3498DB', lw=4, solid_capstyle='round')
    ax.scatter([0.45 + 0.43*fill_ratio], [ypos + 0.01], color='#2980B9', s=70, zorder=5)

ax.text(0.08, 0.38, "Choose Classification Algorithm:", fontsize=10, fontweight='bold', color='#2C3E50')
rect_dd = patches.Rectangle((0.08, 0.28), 0.40, 0.07, facecolor='#ECF0F1', edgecolor='#95A5A6', lw=1)
ax.add_patch(rect_dd)
ax.text(0.10, 0.315, "k-Nearest Neighbors (k=5)  v", fontsize=9, color='#2C3E50')

rect_btn = patches.Rectangle((0.55, 0.27), 0.33, 0.09, facecolor='#E74C3C', edgecolor='#C0392B', lw=1.5)
ax.add_patch(rect_btn)
ax.text(0.715, 0.315, "[ Predict Species ]", color='white', fontsize=10, fontweight='bold', ha='center', va='center')

ax.text(0.5, 0.12, "Ready for prediction. Adjust sliders and click 'Predict Species'.", color='#7F8C8D', fontsize=9, fontstyle='italic', ha='center')

plt.title("Figure 7.8: Iris Flower Classifier Streamlit Application Interface", fontsize=11, fontweight='bold', color='#2C3E50', pad=12)
plt.savefig("figures/fig_7_8_streamlit_app.png", bbox_inches='tight', dpi=200)
plt.close()

# Update fig 7.9 without emoji warnings
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
ax.set_facecolor('#F8F9FA')
ax.axis('off')

rect_bg = patches.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor='white', edgecolor='#BDC3C7', lw=1.5)
ax.add_patch(rect_bg)
header = patches.Rectangle((0.02, 0.88), 0.96, 0.10, facecolor='#2C3E50', edgecolor='none')
ax.add_patch(header)
ax.text(0.5, 0.93, "Iris Multiclass Classification Web App (Streamlit)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')

result_box = patches.Rectangle((0.10, 0.42), 0.80, 0.38, facecolor='#D4EDDA', edgecolor='#28A745', lw=2)
ax.add_patch(result_box)

ax.text(0.5, 0.72, "PREDICTION RESULT", color='#155724', fontsize=13, fontweight='bold', ha='center')
ax.text(0.5, 0.60, "Predicted Species: Iris Setosa", color='#155724', fontsize=15, fontweight='bold', ha='center')
ax.text(0.5, 0.50, "Classification Model: k-Nearest Neighbors (k=5)", color='#155724', fontsize=10, ha='center')
ax.text(0.5, 0.44, "Class Confidence / Probability: 100.0%", color='#155724', fontsize=10, fontweight='bold', ha='center')

ax.text(0.5, 0.28, "Input Features Submitted:", color='#2C3E50', fontsize=10, fontweight='bold', ha='center')
ax.text(0.5, 0.20, "Sepal Length: 5.1 cm  |  Sepal Width: 3.5 cm  |  Petal Length: 1.4 cm  |  Petal Width: 0.2 cm",
        color='#34495E', fontsize=9, ha='center', fontfamily='monospace')

plt.title("Figure 7.9: Final Iris Species Prediction Output in Streamlit", fontsize=11, fontweight='bold', color='#2C3E50', pad=12)
plt.savefig("figures/fig_7_9_prediction_output.png", bbox_inches='tight', dpi=200)
plt.close()
print("Clean images saved.")
