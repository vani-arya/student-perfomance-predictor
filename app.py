"""
app.py — Streamlit Web App
──────────────────────────
Student Performance Predictor UI

Run: python -m streamlit run app.py
"""

import os
import sys
import subprocess
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1f4e79, #2e75b6);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.1rem;
    }
    .score-big {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: 2px;
    }
    .grade-badge {
        font-size: 1.4rem;
        font-weight: 700;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Auto-train if artifacts missing ──────────────────────────
def ensure_artifacts():
    if not os.path.exists("artifacts/model.pkl") or not os.path.exists("artifacts/preprocessor.pkl"):
        st.warning("⚙️ First run detected — training the model now, please wait...")
        with st.spinner("Training models..."):
            result = subprocess.run(
                [sys.executable, "-m", "src.pipeline.train_pipeline"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"Training failed:\n{result.stderr}")
                st.stop()
        st.success("✅ Model trained and saved!")
        st.rerun()


# ── Grade helper ─────────────────────────────────────────────
def score_to_grade(score: float) -> tuple:
    """Returns (grade_letter, color, emoji)"""
    if score >= 90: return "A+", "#2ecc71", "🏆"
    if score >= 80: return "A",  "#27ae60", "🌟"
    if score >= 70: return "B",  "#3498db", "👍"
    if score >= 60: return "C",  "#f39c12", "📚"
    if score >= 50: return "D",  "#e67e22", "⚠️"
    return               "F",  "#e74c3c", "❌"


# ── Gauge chart ──────────────────────────────────────────────
def draw_gauge(score: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)

    # Background arc
    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, [1] * 200, color="#e0e0e0", linewidth=20, solid_capstyle="round")

    # Score arc
    fill_theta = np.linspace(0, np.pi * (score / 100), 200)
    color = "#2ecc71" if score >= 70 else "#f39c12" if score >= 50 else "#e74c3c"
    ax.plot(fill_theta, [1] * len(fill_theta), color=color, linewidth=20, solid_capstyle="round")

    ax.set_ylim(0, 1.5)
    ax.axis("off")
    ax.text(np.pi / 2, -0.3, f"{score:.1f}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color, transform=ax.transData)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    ensure_artifacts()

    # Header
    st.markdown('<div class="main-header">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict math exam scores using Machine Learning</div>', unsafe_allow_html=True)
    st.divider()

    # ── Sidebar — About ──────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
        st.title("About")
        st.markdown("""
        This app predicts a student's **math score** based on:
        - Demographic info
        - Parental education
        - Lunch type
        - Test preparation
        - Reading & writing scores

        **Model:** Best performer from Linear Regression,
        Decision Tree, Random Forest, and Gradient Boosting.

        **Metric:** R² score on test set
        """)
        st.divider()
        if st.button("🔄 Retrain Model"):
            for f in ["artifacts/model.pkl", "artifacts/preprocessor.pkl"]:
                if os.path.exists(f): os.remove(f)
            st.rerun()

    # ── Input Form ───────────────────────────────────────────
    st.subheader("📋 Enter Student Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race   = st.selectbox("Race / Ethnicity", [
            "group A", "group B", "group C", "group D", "group E"
        ])

    with col2:
        parent_edu = st.selectbox("Parental Education Level", [
            "high school", "some college", "associate's degree",
            "bachelor's degree", "master's degree"
        ])
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

    with col3:
        test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
        reading_score = st.slider("Reading Score", 0, 100, 70,
                                   help="Student's reading exam score")
        writing_score = st.slider("Writing Score", 0, 100, 70,
                                   help="Student's writing exam score")

    st.divider()

    # ── Predict Button ───────────────────────────────────────
    if st.button("🚀 Predict Math Score", use_container_width=True, type="primary"):
        try:
            from src.pipeline.predict_pipeline import PredictPipeline, CustomData

            # Build input
            student = CustomData(
                gender=gender,
                race_ethnicity=race,
                parental_education=parent_edu,
                lunch=lunch,
                test_prep_course=test_prep,
                reading_score=reading_score,
                writing_score=writing_score,
            )
            input_df = student.to_dataframe()

            # Predict
            pipeline = PredictPipeline()
            predicted_score = pipeline.predict(input_df)
            predicted_score = max(0, min(100, predicted_score))  # clip to [0,100]

            grade, color, emoji = score_to_grade(predicted_score)

            # ── Results layout ───────────────────────────────
            st.divider()
            st.subheader("📊 Prediction Results")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(f"""
                <div class="result-box">
                    <div>Predicted Math Score</div>
                    <div class="score-big">{predicted_score:.1f}</div>
                    <div>/100</div>
                    <div class="grade-badge" style="background:{color};">
                        {emoji} Grade {grade}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                fig = draw_gauge(predicted_score)
                st.pyplot(fig, use_container_width=True)

            # ── Input Summary Table ──────────────────────────
            st.subheader("📋 Input Summary")
            summary = pd.DataFrame({
                "Feature": [
                    "Gender", "Race/Ethnicity", "Parental Education",
                    "Lunch", "Test Prep", "Reading Score", "Writing Score"
                ],
                "Value": [
                    gender, race, parent_edu, lunch, test_prep,
                    reading_score, writing_score
                ]
            })
            st.table(summary)

            # ── Score Comparison Bar ─────────────────────────
            st.subheader("📈 Score Comparison")
            scores_df = pd.DataFrame({
                "Subject": ["Reading", "Writing", "Math (Predicted)"],
                "Score":   [reading_score, writing_score, predicted_score]
            })
            fig2, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["#4C72B0", "#DD8452", color]
            bars = ax.barh(scores_df["Subject"], scores_df["Score"],
                           color=colors, edgecolor="white", height=0.5)
            for bar, val in zip(bars, scores_df["Score"]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}", va="center", fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Score")
            ax.set_title("Subject Score Comparison")
            ax.axvline(50, color="red", linestyle="--", alpha=0.4, label="Pass mark")
            ax.legend()
            fig2.patch.set_alpha(0)
            st.pyplot(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Make sure you've run the training pipeline first.")

    # ── Footer ───────────────────────────────────────────────
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Python 🐍")


if __name__ == "__main__":
    main()
