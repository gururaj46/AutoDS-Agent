import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agent.planner import detect_problem_type
from utils.preprocessing import preprocess_data
from agent.model_selector import train_and_evaluate


st.set_page_config(
    page_title="AutoDS - AI Data Scientist",
    layout="wide"
)

st.title("🧠 AutoDS - Autonomous Data Scientist Agent")
st.markdown("### 🚀 AI-Powered Automated Machine Learning Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.markdown("---")
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run AutoDS Agent"):

        # =====================
        # Core Pipeline
        # =====================
        problem_type = detect_problem_type(df, target_column)

        X, y = preprocess_data(df.copy(), target_column)

        model, metrics, feature_importance = train_and_evaluate(X, y, problem_type)

        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict()
        }

        # =====================
        # MODEL PERFORMANCE
        # =====================
        st.markdown("---")
        st.subheader("📈 Model Performance")

        perf_col1, perf_col2 = st.columns(2)

        perf_col1.metric("🏆 Best Model", metrics["Best Model"])
        perf_col2.metric("📊 Best Score", metrics["Best Score"])

        comparison_df = pd.DataFrame(
            list(metrics["All Model Scores"].items()),
            columns=["Model", "Score"]
        )

        fig_compare = px.bar(
            comparison_df,
            x="Model",
            y="Score",
            color="Model",
            title="Model Performance Comparison",
            text="Score"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # =====================
        # TARGET DISTRIBUTION
        # =====================
        st.markdown("---")
        st.subheader("📊 Target Distribution")

        if problem_type == "Classification":
            fig_target = px.histogram(
                df,
                x=target_column,
                color=target_column,
                title="Target Class Distribution"
            )
        else:
            fig_target = px.histogram(
                df,
                x=target_column,
                nbins=20,
                title="Target Distribution"
            )

        st.plotly_chart(fig_target, use_container_width=True)

        # =====================
        # CORRELATION HEATMAP
        # =====================
        st.markdown("---")
        st.subheader("🔥 Correlation Heatmap")

        numeric_df = df.select_dtypes(include=["int64", "float64"])

        if not numeric_df.empty:
            corr = numeric_df.corr()

            fig_heatmap = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1
                )
            )

            fig_heatmap.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # =====================
        # FEATURE IMPORTANCE
        # =====================
        if feature_importance:
            st.markdown("---")
            st.subheader("🏆 Feature Importance")

            importance_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=["Feature", "Importance"]
            )

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance Ranking",
                text="Importance"
            )

            st.plotly_chart(fig_importance, use_container_width=True)

        # =====================
        # EXECUTIVE SUMMARY
        # =====================
        st.markdown("---")
        st.subheader("📄 Executive Summary")

        sum_col1, sum_col2 = st.columns(2)

        with sum_col1:
            st.markdown("### 📊 Dataset Info")
            st.markdown(f"- **Rows:** {summary['shape'][0]}")
            st.markdown(f"- **Columns:** {summary['shape'][1]}")
            st.markdown(f"- **Problem Type:** {problem_type}")

        with sum_col2:
            st.markdown("### 🏆 Best Model")
            st.success(f"{metrics['Best Model']} selected as optimal model")
            st.markdown(f"- **Best Score:** {metrics['Best Score']}")

        total_missing = sum(summary['missing_values'].values())
        if total_missing > 0:
            st.warning(f"⚠️ Dataset contains {total_missing} missing values.")
        else:
            st.success("✅ No missing values detected.")

        st.markdown("---")

        if feature_importance:
            st.markdown("### 🔎 Key Business Drivers")
            top_features = list(feature_importance.items())[:3]

            for feature, importance in top_features:
                st.markdown(f"- **{feature}** → Importance Score: `{importance}`")

        st.markdown("---")
        st.info("💡 Recommendation: Consider hyperparameter tuning and advanced feature engineering to further improve performance.")
