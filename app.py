import datetime
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Iris AI Studio | Multiclass Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern AI Analytics Dashboard
st.markdown("""
<style>
    /* Google Fonts & Base Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header Card */
    .studio-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);
    }
    .studio-header h1 {
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .studio-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.95;
    }
    
    /* Stat Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    
    /* Model Performance Cards */
    .model-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        position: relative;
    }
    .model-card.featured {
        border: 2px solid #6366F1;
        background: linear-gradient(180deg, #FFFFFF 0%, #EEF2FF 100%);
    }
    .badge-top {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #4F46E5;
        color: white;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prediction Banner */
    .prediction-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #86EFAC;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: #14532D;
        margin-top: 1.5rem;
    }
    
    /* Workflow Steps */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        margin: 1.5rem 0;
        overflow-x: auto;
    }
    .workflow-step {
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: #334155;
    }
    .workflow-step .step-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
        display: block;
    }
    .workflow-arrow {
        color: #94A3B8;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    /* Section Subheaders */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data
def get_iris_dataset():
    """Load Iris dataset and structure into train/test splits identically to training."""
    raw = load_iris()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['target'] = raw.target
    df['species'] = pd.Categorical.from_codes(raw.target, ['Setosa', 'Versicolor', 'Virginica'])
    
    X = raw.data
    y = raw.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return {
        'df': df,
        'raw': raw,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': raw.feature_names,
        'target_names': ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    }

@st.cache_resource
def load_models():
    """Load pre-trained models safely with fallback warnings."""
    model_paths = {
        'Logistic Regression': 'logistic_model.pkl',
        'kNN (k=5)': 'knn_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl'
    }
    loaded = {}
    errors = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception as e:
                errors[name] = f"Error loading {path}: {str(e)}"
        else:
            errors[name] = f"File {path} not found in workspace."
    return loaded, errors

dataset_info = get_iris_dataset()
models, model_errors = load_models()

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize session state for slider values
if 'sl_val' not in st.session_state:
    st.session_state.sl_val = 5.1
if 'sw_val' not in st.session_state:
    st.session_state.sw_val = 3.5
if 'pl_val' not in st.session_state:
    st.session_state.pl_val = 1.4
if 'pw_val' not in st.session_state:
    st.session_state.pw_val = 0.2

# Helper to load example inputs
def set_sample_values(sl, sw, pl, pw):
    st.session_state.sl_val = float(sl)
    st.session_state.sw_val = float(sw)
    st.session_state.pl_val = float(pl)
    st.session_state.pw_val = float(pw)

# ==============================================================================
# 3. SIDEBAR NAVIGATION & APP BRANDING
# ==============================================================================
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=75)
    
    st.markdown("### **IRIS AI STUDIO**")
    st.caption("Multiclass Classification & Model Analysis Platform")
    st.write("---")
    
    menu_selection = st.radio(
        "Navigation Menu",
        [
            "📊 Home Dashboard",
            "🔮 Predict Iris Species",
            "⚖️ Model Comparison & Evaluation",
            "🔍 Iris Dataset Explorer",
            "📜 Prediction History"
        ],
        index=0
    )
    
    st.write("---")
    st.markdown("#### ⚡ System Status")
    all_models_ready = len(models) == 3
    if all_models_ready:
        st.success("● 3 Models Loaded & Active", icon="✅")
    else:
        st.warning(f"● {len(models)}/3 Models Available", icon="⚠️")
        for m_name, err in model_errors.items():
            st.caption(f"- {m_name}: {err}")
            
    st.write("---")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #64748B; line-height: 1.4;'>
        <strong>GenAI Model Making Project</strong><br/>
        Topic 31 | Group 16 | Class: LYCORE 2<br/>
        MIT School of Computing, Pune<br/>
        Academic Year: 2026–27
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION ONE — HOME DASHBOARD
# ==============================================================================
if menu_selection == "📊 Home Dashboard":
    st.markdown("""
    <div class="studio-header">
        <h1>🌸 IRIS AI STUDIO</h1>
        <p>Multiclass Classification &amp; Model Analysis Platform — Enterprise AI/ML Jury Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Total Dataset Samples</div>
            <div class="kpi-value">150</div>
            <div class="kpi-sub">50 samples per class</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Input Features</div>
            <div class="kpi-value">4</div>
            <div class="kpi-sub">Sepal &amp; Petal measurements</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Iris Species</div>
            <div class="kpi-value">3</div>
            <div class="kpi-sub">Setosa, Versicolor, Virginica</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">ML Algorithms</div>
            <div class="kpi-value">3</div>
            <div class="kpi-sub">LR, kNN, Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div class='section-title'>🎯 Model Performance Overview</div>", unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="model-card">
            <h3 style="margin-top:0; color:#1E293B; font-weight:700;">Logistic Regression</h3>
            <p style="color:#64748B; font-size:0.9rem;">Probabilistic linear multiclass classifier</p>
            <div style="font-size:2rem; font-weight:800; color:#2563EB;">96.67%</div>
            <p style="color:#475569; font-weight:600; margin-top:0.3rem;">Test Accuracy: 29 / 30 correct</p>
            <hr style="border:none; border-top:1px solid #E2E8F0; margin:1rem 0;"/>
            <span style="font-size:0.8rem; color:#64748B;">max_iter=200 | Solved via lbfgs</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown("""
        <div class="model-card featured">
            <div class="badge-top">Top Performer</div>
            <h3 style="margin-top:0; color:#1E293B; font-weight:700;">k-Nearest Neighbors</h3>
            <p style="color:#64748B; font-size:0.9rem;">Distance-based instance classification (k=5)</p>
            <div style="font-size:2rem; font-weight:800; color:#059669;">100.00%</div>
            <p style="color:#047857; font-weight:600; margin-top:0.3rem;">Test Accuracy: 30 / 30 correct</p>
            <hr style="border:none; border-top:1px solid #C7D2FE; margin:1rem 0;"/>
            <span style="font-size:0.8rem; color:#4338CA; font-weight:600;">Highest performance on test split</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown("""
        <div class="model-card">
            <h3 style="margin-top:0; color:#1E293B; font-weight:700;">Decision Tree</h3>
            <p style="color:#64748B; font-size:0.9rem;">Recursive feature-rule hierarchical split</p>
            <div style="font-size:2rem; font-weight:800; color:#D97706;">93.33%</div>
            <p style="color:#475569; font-weight:600; margin-top:0.3rem;">Test Accuracy: 28 / 30 correct</p>
            <hr style="border:none; border-top:1px solid #E2E8F0; margin:1rem 0;"/>
            <span style="font-size:0.8rem; color:#64748B;">random_state=42 | Axis-aligned splits</span>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Visualizations
    st.markdown("<div class='section-title'>📈 Visual Analytics Preview</div>", unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        # Accuracy Bar Chart
        acc_df = pd.DataFrame({
            'Model': ['kNN (k=5)', 'Logistic Regression', 'Decision Tree'],
            'Accuracy': [100.00, 96.67, 93.33],
            'Color': ['#10B981', '#3B82F6', '#F59E0B']
        })
        fig_acc = px.bar(
            acc_df, x='Model', y='Accuracy', text='Accuracy',
            color='Model',
            color_discrete_map={'kNN (k=5)': '#10B981', 'Logistic Regression': '#3B82F6', 'Decision Tree': '#F59E0B'},
            title="<b>Model Accuracy Comparison (Test Dataset)</b>"
        )
        fig_acc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_acc.update_layout(yaxis_range=[85, 105], showlegend=False, height=350, margin=dict(t=50, b=40, l=40, r=40))
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with c_chart2:
        # Species Distribution Pie
        species_counts = dataset_info['df']['species'].value_counts().reset_index()
        species_counts.columns = ['Species', 'Count']
        fig_pie = px.pie(
            species_counts, names='Species', values='Count',
            color='Species',
            color_discrete_map={'Setosa': '#8B5CF6', 'Versicolor': '#3B82F6', 'Virginica': '#EC4899'},
            title="<b>Iris Class Balance (50 samples per species)</b>",
            hole=0.45
        )
        fig_pie.update_layout(height=350, margin=dict(t=50, b=40, l=40, r=40))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Workflow Section
    st.markdown("<div class='section-title'>🔄 End-to-End ML Pipeline Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">
            <span class="step-icon">📂</span>
            1. Iris Dataset<br/><span style="font-size:0.75rem; color:#64748B;">150 Botanical Samples</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="step-icon">⚙️</span>
            2. Preprocessing<br/><span style="font-size:0.75rem; color:#64748B;">Stratified 80/20 Split</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="step-icon">🧠</span>
            3. Model Training<br/><span style="font-size:0.75rem; color:#64748B;">LR, kNN, Decision Tree</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="step-icon">📊</span>
            4. Model Evaluation<br/><span style="font-size:0.75rem; color:#64748B;">Accuracy &amp; Confusion Matrix</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="step-icon">💾</span>
            5. Model Persistence<br/><span style="font-size:0.75rem; color:#64748B;">Joblib Serialization (.pkl)</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="step-icon">🌐</span>
            6. Real-time Inference<br/><span style="font-size:0.75rem; color:#64748B;">Iris AI Studio Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    with st.expander("ℹ️ About the Project & Academic Framework", expanded=True):
        st.markdown("""
        **Iris AI Studio** is an advanced machine learning classification and diagnostic platform engineered for the **GenAI Model Making Project (Topic 31)** at **MIT School of Computing, Pune**.
        
        - **Objective:** Establish an automated botanical species classification engine comparing linear, distance-based, and decision-tree inductive biases on continuous flower morphology features.
        - **Dataset Quality:** The benchmark Iris dataset consists of 150 instances of Iris flowers across three balanced classes (Setosa, Versicolor, and Virginica) with 4 numerical features measured in centimeters.
        - **Model Deployment:** Models are trained using Scikit-Learn under a strict 80% train / 20% test stratified split (`random_state=42`) and persisted via Joblib binaries.
        """)


# ==============================================================================
# SECTION TWO — PREDICT IRIS SPECIES
# ==============================================================================
elif menu_selection == "🔮 Predict Iris Species":
    st.markdown("""
    <div class="studio-header">
        <h1>🔮 Real-Time Iris Species Prediction</h1>
        <p>Interactive feature input, multiclass probability estimation, and multi-model consensus comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample Presets Row
    st.markdown("#### ⚡ Quick Presets (Benchmark Samples)")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🌸 Load Setosa Sample [5.1, 3.5, 1.4, 0.2]", use_container_width=True):
            set_sample_values(5.1, 3.5, 1.4, 0.2)
            st.rerun()
    with p2:
        if st.button("🌿 Load Versicolor Sample [6.0, 2.9, 4.5, 1.5]", use_container_width=True):
            set_sample_values(6.0, 2.9, 4.5, 1.5)
            st.rerun()
    with p3:
        if st.button("🌺 Load Virginica Sample [6.5, 3.0, 5.5, 1.8]", use_container_width=True):
            set_sample_values(6.5, 3.0, 5.5, 1.8)
            st.rerun()

    st.write("")
    
    # Main Form Area
    col_input, col_config = st.columns([1.8, 1.2])
    
    with col_input:
        st.markdown("#### 📏 Botanical Measurements (Input Features)")
        
        sl = st.slider(
            "Sepal Length (cm)",
            min_value=4.0, max_value=8.0,
            value=st.session_state.sl_val,
            step=0.1,
            help="Length of the flower sepal in centimeters."
        )
        sw = st.slider(
            "Sepal Width (cm)",
            min_value=2.0, max_value=4.5,
            value=st.session_state.sw_val,
            step=0.1,
            help="Width of the flower sepal in centimeters."
        )
        pl = st.slider(
            "Petal Length (cm)",
            min_value=1.0, max_value=7.0,
            value=st.session_state.pl_val,
            step=0.1,
            help="Length of the flower petal in centimeters."
        )
        pw = st.slider(
            "Petal Width (cm)",
            min_value=0.1, max_value=2.5,
            value=st.session_state.pw_val,
            step=0.1,
            help="Width of the flower petal in centimeters."
        )

    with col_config:
        st.markdown("#### ⚙️ Inference Configuration")
        model_choice = st.selectbox(
            "Select Classification Algorithm:",
            [
                "kNN (k=5)",
                "Logistic Regression",
                "Decision Tree",
                "Compare All Models"
            ],
            index=0,
            help="Choose an individual pre-trained model or benchmark all 3 simultaneously."
        )
        
        st.markdown("""
        <div style="background:#F1F5F9; border-radius:12px; padding:1rem; font-size:0.85rem; color:#475569; margin-top:1rem;">
            <strong>Input Summary Vector:</strong><br/>
            <code>[Sepal L: {:.1f}, Sepal W: {:.1f}, Petal L: {:.1f}, Petal W: {:.1f}]</code>
        </div>
        """.format(sl, sw, pl, pw), unsafe_allow_html=True)
        
        predict_clicked = st.button("🚀 PREDICT IRIS SPECIES", type="primary", use_container_width=True)

    # Prediction Execution Logic
    if predict_clicked:
        input_vector = np.array([[sl, sw, pl, pw]], dtype=float)
        species_labels = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
        species_colors = {"Iris Setosa": "#8B5CF6", "Iris Versicolor": "#3B82F6", "Iris Virginica": "#EC4899"}
        
        st.write("---")
        
        # Scenario A: Individual Model Prediction
        if model_choice != "Compare All Models":
            if model_choice not in models:
                st.error(f"Selected model '{model_choice}' is not available. Please check model files.")
            else:
                active_model = models[model_choice]
                pred_idx = int(active_model.predict(input_vector)[0])
                predicted_name = species_labels[pred_idx]
                
                # Probability check
                probs = active_model.predict_proba(input_vector)[0] if hasattr(active_model, 'predict_proba') else None
                top_prob = float(probs[pred_idx] * 100) if probs is not None else 100.0
                
                # Append to session history
                st.session_state.prediction_history.append({
                    'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Sepal Length': sl,
                    'Sepal Width': sw,
                    'Petal Length': pl,
                    'Petal Width': pw,
                    'Selected Model': model_choice,
                    'Predicted Species': predicted_name,
                    'Confidence': f"{top_prob:.1f}%"
                })
                
                # Result Card Display
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                        <div>
                            <span style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#15803D;">Classification Result</span>
                            <h2 style="margin:0.2rem 0; font-size:2.2rem; font-weight:800; color:#14532D;">🌺 {predicted_name}</h2>
                            <p style="margin:0; font-size:0.95rem; color:#166534;">
                                Classified via <strong>{model_choice}</strong> &bull; Estimated Probability: <strong>{top_prob:.1f}%</strong>
                            </p>
                        </div>
                        <div style="background:#FFFFFF; border:1px solid #BBF7D0; border-radius:12px; padding:0.8rem 1.4rem; text-align:right;">
                            <span style="font-size:0.75rem; color:#64748B; font-weight:600;">Status</span><br/>
                            <span style="font-weight:700; color:#059669; font-size:1.1rem;">Inference Complete ✅</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability Chart
                if probs is not None:
                    st.markdown("<div class='section-title'>📊 Estimated Class Probability Distribution</div>", unsafe_allow_html=True)
                    st.caption("⚠️ Note: Probabilities represent normalized softmax/frequency scores learned from training distributions, not guaranteed certainty.")
                    
                    prob_df = pd.DataFrame({
                        'Species': ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'],
                        'Probability (%)': [p * 100 for p in probs]
                    })
                    fig_prob = px.bar(
                        prob_df, x='Probability (%)', y='Species', orientation='h',
                        color='Species',
                        color_discrete_map=species_colors,
                        text='Probability (%)'
                    )
                    fig_prob.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig_prob.update_layout(xaxis_range=[0, 115], height=240, showlegend=False, margin=dict(t=20, b=30, l=40, r=40))
                    st.plotly_chart(fig_prob, use_container_width=True)

        # Scenario B: Compare All Models
        else:
            st.markdown("<div class='section-title'>⚖️ Multi-Model Consensus & Comparison</div>", unsafe_allow_html=True)
            
            comp_results = []
            prob_breakdown = {}
            
            for m_name in ['kNN (k=5)', 'Logistic Regression', 'Decision Tree']:
                if m_name in models:
                    m_obj = models[m_name]
                    p_idx = int(m_obj.predict(input_vector)[0])
                    p_name = species_labels[p_idx]
                    p_probs = m_obj.predict_proba(input_vector)[0] if hasattr(m_obj, 'predict_proba') else None
                    p_top_prob = f"{p_probs[p_idx]*100:.1f}%" if p_probs is not None else "N/A"
                    
                    comp_results.append({
                        'Model': m_name,
                        'Predicted Species': p_name,
                        'Class Confidence': p_top_prob
                    })
                    prob_breakdown[m_name] = p_probs
                    
                    # Record comparison runs
                    st.session_state.prediction_history.append({
                        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sepal Length': sl,
                        'Sepal Width': sw,
                        'Petal Length': pl,
                        'Petal Width': pw,
                        'Selected Model': m_name,
                        'Predicted Species': p_name,
                        'Confidence': p_top_prob
                    })

            comp_df = pd.DataFrame(comp_results)
            
            # Check Agreement
            distinct_predictions = comp_df['Predicted Species'].nunique()
            if distinct_predictions == 1:
                consensus_species = comp_df['Predicted Species'].iloc[0]
                st.success(f"🤝 **Consensus Reached:** All 3 models unanimously agree that the sample is **{consensus_species}**!", icon="✨")
            else:
                st.warning(f"⚠️ **Model Disagreement:** Models predicted {distinct_predictions} distinct species for this input sample. Review decision boundary characteristics.", icon="⚡")
                
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Detailed Probability Comparison Table
            st.markdown("#### 🔬 Detailed Model Probability Distributions")
            prob_comp_rows = []
            for m_name, p_vals in prob_breakdown.items():
                if p_vals is not None:
                    prob_comp_rows.append({
                        'Model': m_name,
                        'Iris Setosa (%)': f"{p_vals[0]*100:.2f}%",
                        'Iris Versicolor (%)': f"{p_vals[1]*100:.2f}%",
                        'Iris Virginica (%)': f"{p_vals[2]*100:.2f}%"
                    })
            st.table(pd.DataFrame(prob_comp_rows))


# ==============================================================================
# SECTION THREE — MODEL COMPARISON & EVALUATION
# ==============================================================================
elif menu_selection == "⚖️ Model Comparison & Evaluation":
    st.markdown("""
    <div class="studio-header">
        <h1>⚖️ Model Comparison &amp; In-Depth Evaluation</h1>
        <p>Comprehensive benchmarking of Logistic Regression, k-Nearest Neighbors, and Decision Tree</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Performance Overview Table
    st.markdown("<div class='section-title'>🏆 Overall Model Accuracy Benchmark</div>", unsafe_allow_html=True)
    
    perf_summary = pd.DataFrame({
        'Model': ['k-Nearest Neighbors (k=5)', 'Logistic Regression', 'Decision Tree'],
        'Algorithm Family': ['Instance / Distance-based', 'Generalized Linear Model', 'Hierarchical Decision Rules'],
        'Test Accuracy': ['100.00%', '96.67%', '93.33%'],
        'Correct Predictions': ['30 / 30', '29 / 30', '28 / 30'],
        'Misclassifications': ['0', '1', '2']
    })
    st.dataframe(perf_summary, use_container_width=True, hide_index=True)
    
    # 2. Detailed Classification Metrics Computed from Actual Test Split
    st.markdown("<div class='section-title'>📐 Rigorous Evaluation Metrics (Actual Test Split)</div>", unsafe_allow_html=True)
    st.caption("Metrics computed directly from the stratified 20% test split (30 samples: 10 Setosa, 10 Versicolor, 10 Virginica).")
    
    X_test = dataset_info['X_test']
    y_test = dataset_info['y_test']
    
    tab_lr, tab_knn, tab_dt = st.tabs(["Logistic Regression", "kNN (k=5)", "Decision Tree"])
    
    def generate_metric_table(model_obj):
        y_pred = model_obj.predict(X_test)
        rep = classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica'], output_dict=True)
        m_df = pd.DataFrame({
            'Class / Metric': ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica', 'Macro Average', 'Weighted Average'],
            'Precision': [f"{rep['Setosa']['precision']:.4f}", f"{rep['Versicolor']['precision']:.4f}", f"{rep['Virginica']['precision']:.4f}", f"{rep['macro avg']['precision']:.4f}", f"{rep['weighted avg']['precision']:.4f}"],
            'Recall': [f"{rep['Setosa']['recall']:.4f}", f"{rep['Versicolor']['recall']:.4f}", f"{rep['Virginica']['recall']:.4f}", f"{rep['macro avg']['recall']:.4f}", f"{rep['weighted avg']['recall']:.4f}"],
            'F1-Score': [f"{rep['Setosa']['f1-score']:.4f}", f"{rep['Versicolor']['f1-score']:.4f}", f"{rep['Virginica']['f1-score']:.4f}", f"{rep['macro avg']['f1-score']:.4f}", f"{rep['weighted avg']['f1-score']:.4f}"],
            'Support': [int(rep['Setosa']['support']), int(rep['Versicolor']['support']), int(rep['Virginica']['support']), int(rep['macro avg']['support']), int(rep['weighted avg']['support'])]
        })
        return m_df, confusion_matrix(y_test, y_pred)
    
    with tab_lr:
        if 'Logistic Regression' in models:
            lr_metrics, lr_cm = generate_metric_table(models['Logistic Regression'])
            st.dataframe(lr_metrics, use_container_width=True, hide_index=True)
        else:
            st.error("Logistic Regression model file not loaded.")
            
    with tab_knn:
        if 'kNN (k=5)' in models:
            knn_metrics, knn_cm = generate_metric_table(models['kNN (k=5)'])
            st.dataframe(knn_metrics, use_container_width=True, hide_index=True)
        else:
            st.error("kNN model file not loaded.")
            
    with tab_dt:
        if 'Decision Tree' in models:
            dt_metrics, dt_cm = generate_metric_table(models['Decision Tree'])
            st.dataframe(dt_metrics, use_container_width=True, hide_index=True)
        else:
            st.error("Decision Tree model file not loaded.")

    # 3. Interactive Confusion Matrices
    st.markdown("<div class='section-title'>🎯 Confusion Matrix Heatmaps</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    def plot_cm(cm, title, color_scale):
        fig = px.imshow(
            cm, text_auto=True,
            x=class_names, y=class_names,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            color_continuous_scale=color_scale,
            title=f"<b>{title}</b>"
        )
        fig.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20), coloraxis_showscale=False)
        return fig

    with c1:
        if 'Logistic Regression' in models:
            st.plotly_chart(plot_cm(lr_cm, "Logistic Regression (96.67%)", "Blues"), use_container_width=True)
    with c2:
        if 'kNN (k=5)' in models:
            st.plotly_chart(plot_cm(knn_cm, "kNN k=5 (100.00%)", "Greens"), use_container_width=True)
    with c3:
        if 'Decision Tree' in models:
            st.plotly_chart(plot_cm(dt_cm, "Decision Tree (93.33%)", "Oranges"), use_container_width=True)

    # 4. Model Explanations & Selection Insights
    st.markdown("<div class='section-title'>💡 Theoretical Model Insights & Algorithmic Analysis</div>", unsafe_allow_html=True)
    
    st.markdown("""
    - **Logistic Regression:** A supervised generalized linear classifier using the multinomial cross-entropy loss function. It maps continuous feature combinations into discrete class probabilities via softmax scoring. It delivers smooth, linear classification boundaries.
    - **k-Nearest Neighbors (k=5):** An instance-based non-parametric classifier. Predictions are governed strictly by local Euclidean geometry among the $k$ closest neighbors in feature space. Because the Iris test split forms dense, well-isolated clusters, kNN yielded 100% accuracy.
    - **Decision Tree (CART):** Constructs orthogonal axis-aligned splitting hyperplanes based on Gini impurity. While highly interpretable, standard trees can experience slight boundary rigidity near dense feature overlaps between Versicolor and Virginica.
    
    > **Academic Jury Note on Performance:** Although kNN achieved a perfect 100.00% accuracy score on our 30-sample test set, this does not imply that kNN is universally optimal for all unseen datasets. On larger, noisier, or higher-dimensional datasets, distance metrics suffer from the *curse of dimensionality* and higher latency during inference.
    """)


# ==============================================================================
# SECTION FOUR — DATASET EXPLORER
# ==============================================================================
elif menu_selection == "🔍 Iris Dataset Explorer":
    st.markdown("""
    <div class="studio-header">
        <h1>🔍 Iris Dataset Explorer</h1>
        <p>Interactive exploratory data analysis, bivariate scatter plots, and feature correlation matrix</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = dataset_info['df']
    
    # Overview metrics
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Total Observations", len(df))
    with d2:
        st.metric("Total Features", 4)
    with d3:
        st.metric("Target Classes", "3 Species")
    with d4:
        st.metric("Missing Values", "0 (Clean)")

    # Data Filter
    st.markdown("<div class='section-title'>📋 Interactive Dataset Preview</div>", unsafe_allow_html=True)
    species_filter = st.selectbox(
        "Filter by Species:",
        ["All Species", "Setosa", "Versicolor", "Virginica"],
        index=0
    )
    
    if species_filter == "All Species":
        filtered_df = df
    else:
        filtered_df = df[df['species'] == species_filter]
        
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Statistics Summary
    st.markdown("<div class='section-title'>📊 Summary Statistics (Selected View)</div>", unsafe_allow_html=True)
    feature_cols = dataset_info['feature_names']
    stats_df = filtered_df[feature_cols].describe().T[['mean', 'std', 'min', 'max']].reset_index()
    stats_df.columns = ['Feature Name', 'Mean (cm)', 'Std Dev', 'Min (cm)', 'Max (cm)']
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown("<div class='section-title'>📈 Bivariate & Correlation Analysis</div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    
    with v1:
        # Sepal Scatter
        fig_sepal = px.scatter(
            df, x='sepal length (cm)', y='sepal width (cm)',
            color='species',
            color_discrete_map={'Setosa': '#8B5CF6', 'Versicolor': '#3B82F6', 'Virginica': '#EC4899'},
            title="<b>Sepal Length vs Sepal Width</b>",
            labels={'sepal length (cm)': 'Sepal Length (cm)', 'sepal width (cm)': 'Sepal Width (cm)'}
        )
        fig_sepal.update_layout(height=360)
        st.plotly_chart(fig_sepal, use_container_width=True)
        
    with v2:
        # Petal Scatter
        fig_petal = px.scatter(
            df, x='petal length (cm)', y='petal width (cm)',
            color='species',
            color_discrete_map={'Setosa': '#8B5CF6', 'Versicolor': '#3B82F6', 'Virginica': '#EC4899'},
            title="<b>Petal Length vs Petal Width (Strongest Separation)</b>",
            labels={'petal length (cm)': 'Petal Length (cm)', 'petal width (cm)': 'Petal Width (cm)'}
        )
        fig_petal.update_layout(height=360)
        st.plotly_chart(fig_petal, use_container_width=True)
        
    # Correlation Matrix Heatmap
    corr_matrix = df[feature_cols].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=True,
        color_continuous_scale="Purples",
        title="<b>Pearson Correlation Heatmap Between Features</b>",
        labels=dict(color="Correlation")
    )
    fig_corr.update_layout(height=380)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Dataset Download Button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Iris Dataset as CSV",
        data=csv_data,
        file_name="iris_dataset.csv",
        mime="text/csv",
        type="primary"
    )


# ==============================================================================
# SECTION FIVE — PREDICTION HISTORY
# ==============================================================================
elif menu_selection == "📜 Prediction History":
    st.markdown("""
    <div class="studio-header">
        <h1>📜 Real-Time Prediction History &amp; Audit Trail</h1>
        <p>Session-based record of flower measurements, deployed algorithms, and inference outcomes</p>
    </div>
    """, unsafe_allow_html=True)
    
    history_records = st.session_state.prediction_history
    
    # Summary Metrics
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.metric("Total Predictions in Session", len(history_records))
    with h_col2:
        num_models_used = pd.DataFrame(history_records)['Selected Model'].nunique() if history_records else 0
        st.metric("Distinct Models Queried", num_models_used)
    with h_col3:
        st.metric("Storage Mode", "Streamlit Session State")
        
    if not history_records:
        st.info("ℹ️ No predictions have been recorded yet in this session. Head over to **Predict Iris Species** to run inference!")
    else:
        history_df = pd.DataFrame(history_records)
        
        # Filtering Controls
        st.markdown("<div class='section-title'>🔍 Filter History Records</div>", unsafe_allow_html=True)
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            all_species = ["All"] + list(history_df['Predicted Species'].unique())
            filter_sp = st.selectbox("Filter by Predicted Species:", all_species)
        with f_col2:
            all_models = ["All"] + list(history_df['Selected Model'].unique())
            filter_md = st.selectbox("Filter by Algorithm:", all_models)
            
        filtered_history = history_df.copy()
        if filter_sp != "All":
            filtered_history = filtered_history[filtered_history['Predicted Species'] == filter_sp]
        if filter_md != "All":
            filtered_history = filtered_history[filtered_history['Selected Model'] == filter_md]
            
        st.dataframe(filtered_history, use_container_width=True, hide_index=True)
        
        # Export and Clear Buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            history_csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export History to CSV",
                data=history_csv,
                file_name=f"iris_prediction_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        with btn_col2:
            if st.button("🗑️ Clear Session History", type="secondary", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

st.write("---")
st.caption("Iris AI Studio | Developed for Final GenAI Jury Evaluation | MIT School of Computing, ADT University, Pune (2026–27)")
