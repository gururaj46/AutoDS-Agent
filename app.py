import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from agent.planner import detect_problem_type
from utils.preprocessing import preprocess_data
from agent.model_selector import train_and_evaluate


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AutoDS - AI Data Scientist",
    layout="wide"
)

st.title("🧠 AutoDS - Autonomous Data Scientist Agent")
st.markdown("### 🚀 AI-Powered Automated Machine Learning Dashboard")


# ==========================
# SESSION STATE INIT
# ==========================
if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================
# SIDEBAR
# ==========================
fast_mode = st.sidebar.toggle(
    "⚡ Enable Fast Mode (Recommended for large datasets)",
    value=True
)


# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


# ==========================
# CACHED FUNCTIONS
# ==========================
@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def preprocess_cached(df, target_column):
    return preprocess_data(df.copy(), target_column)


# ==========================
# AI REPORT GENERATOR
# ==========================
def generate_ai_insights(problem_type, best_model_name, best_score,
                         leaderboard, feature_importance, df):

    insights = []

    insights.append("EXECUTIVE AI ANALYSIS REPORT\n")
    insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    insights.append(f"Detected problem type: {problem_type}")
    insights.append(f"Best model: {best_model_name} with score {round(best_score,4)}")

    if len(leaderboard) > 1:
        gap = round(best_score - leaderboard.iloc[1]["Score"], 4)
        insights.append(f"Model outperformed second best by {gap}")

    insights.append("\nSTRATEGIC RECOMMENDATIONS:")
    insights.append("- Monitor model performance")
    insights.append("- Retrain periodically")
    insights.append("- Optimize top influencing features")

    return "\n".join(insights)


# ==========================
# PDF GENERATOR
# ==========================
def generate_pdf_report(report_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    style = styles["Normal"]

    for line in report_text.split("\n"):
        elements.append(Paragraph(line, style))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ==========================
# MAIN APP
# ==========================
if uploaded_file is not None:

    df = load_data(uploaded_file)

    st.markdown("---")
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(), width="stretch")

    st.markdown("---")
    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run AutoDS Agent"):

        progress = st.progress(0)
        status = st.empty()

        try:
            status.text("🔎 Detecting problem type...")
            progress.progress(20)
            problem_type = detect_problem_type(df, target_column)

            status.text("🛠 Preprocessing data...")
            progress.progress(40)
            X, y = preprocess_cached(df, target_column)

            status.text("🤖 Training models...")
            progress.progress(70)

            best_model_name, best_score, leaderboard, feature_importance, best_model = train_and_evaluate(
                X, y, problem_type, fast_mode=fast_mode
            )

            progress.progress(100)
            status.text("✅ Analysis Complete!")

            st.session_state.model_results = {
                "problem_type": problem_type,
                "best_model_name": best_model_name,
                "best_score": best_score,
                "leaderboard": leaderboard,
                "feature_importance": feature_importance,
                "df": df,
                "target": target_column
            }

        except ValueError as e:
            st.error(str(e))
            st.stop()


# ==========================
# DISPLAY RESULTS
# ==========================
if st.session_state.model_results:

    results = st.session_state.model_results
    df = results["df"]

    # Leaderboard
    st.markdown("---")
    st.subheader("🏆 Model Leaderboard")
    st.dataframe(results["leaderboard"], width="stretch")
    st.success(f"🥇 Best Model: {results['best_model_name']}")
    st.metric("Best Score", round(results["best_score"], 4))

    # Performance Graph
    st.markdown("---")
    st.subheader("📈 Model Performance Comparison")

    fig_perf = px.bar(
        results["leaderboard"],
        x="Model",
        y="Score",
        text="Score"
    )
    st.plotly_chart(fig_perf, width="stretch")

    # Target Distribution
    st.markdown("---")
    st.subheader("📊 Target Distribution")

    fig_target = px.histogram(df, x=results["target"])
    st.plotly_chart(fig_target, width="stretch")

    # Correlation Heatmap
    st.markdown("---")
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        fig_heatmap = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig_heatmap, width="stretch")
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # Executive Summary
    st.markdown("---")
    st.subheader("📄 Executive Summary")

    colA, colB, colC = st.columns(3)
    colA.metric("Dataset Rows", df.shape[0])
    colB.metric("Dataset Columns", df.shape[1])
    colC.metric("Best Score", round(results["best_score"], 4))

    st.success(f"🏆 Best Model: {results['best_model_name']}")
    st.info(f"🧠 Problem Type: {results['problem_type']}")

    # AI REPORT
    st.markdown("---")
    st.subheader("🤖 AI Business Insight Generator")

    if st.button("🧠 Generate Executive AI Report"):
        st.session_state.ai_report = generate_ai_insights(
            results["problem_type"],
            results["best_model_name"],
            results["best_score"],
            results["leaderboard"],
            results["feature_importance"],
            df
        )

    if st.session_state.ai_report:
        st.text_area("Executive AI Report",
                     st.session_state.ai_report,
                     height=300)

        pdf_buffer = generate_pdf_report(st.session_state.ai_report)

        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_buffer,
            file_name="AutoDS_Executive_Report.pdf",
            mime="application/pdf"
        )

    # ==========================
    # 🤖 AI ML COPILOT CHAT
    # ==========================
    st.markdown("---")
    st.subheader("🧠 AutoDS AI Copilot")

    user_input = st.text_input("Ask your AI ML Copilot:")

    if user_input:

        response = ""

        if "why" in user_input.lower() and "model" in user_input.lower():
            response = f"{results['best_model_name']} was selected because it achieved the highest score of {results['best_score']}."

        elif "roc" in user_input.lower():
            response = "ROC Curve measures classification performance across thresholds. Higher AUC means better separation."

        elif "feature" in user_input.lower():
            if results["feature_importance"]:
                top_feature = list(results["feature_importance"].keys())[0]
                response = f"The most important feature impacting predictions is: {top_feature}"
            else:
                response = "Feature importance not available for this model."

        elif "improve" in user_input.lower():
            response = "To improve performance: add better features, tune hyperparameters, handle imbalance, or increase training data."

        else:
            response = "You can ask about model selection, ROC curve, features, or improvement suggestions."

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Copilot", response))

    for sender, message in st.session_state.chat_history:
        st.write(f"**{sender}:** {message}")


# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<center>© 2026 AutoDS | Developed By : Gururaj Tandur</center>",
    unsafe_allow_html=True
)
