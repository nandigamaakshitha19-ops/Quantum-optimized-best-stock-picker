import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import os

st.set_page_config(page_title="Quantum vs Classical Portfolio", layout="wide")

st.title("⚛️ Quantum vs Classical Portfolio Optimization Engine")
st.markdown("---")

# -------------------------------------------------
# STOCKS
# -------------------------------------------------

tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]

data = yf.download(
    tickers,
    period="1y",
    auto_adjust=True,
    progress=False
)["Close"]

data.dropna(inplace=True)

returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# -------------------------------------------------
# CLASSICAL OPTIMIZATION (Sharpe Max)
# -------------------------------------------------

def neg_sharpe(weights):
    port_return = np.sum(mean_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_vol

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(len(tickers)))
init_guess = np.ones(len(tickers)) / len(tickers)

classical = minimize(neg_sharpe, init_guess, method="SLSQP",
                     bounds=bounds, constraints=constraints)

c_weights = classical.x

# -------------------------------------------------
# QUANTUM-INSPIRED OPTIMIZATION (Random Search)
# -------------------------------------------------

np.random.seed(42)
best_score = -999
q_weights = None
scores = []

for i in range(5000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    ret = np.sum(mean_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    score = ret / vol

    scores.append(score)

    if score > best_score:
        best_score = score
        q_weights = weights

# -------------------------------------------------
# METRICS
# -------------------------------------------------

def portfolio_metrics(weights):
    ret = np.sum(mean_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol
    return ret, vol, sharpe

c_ret, c_vol, c_sharpe = portfolio_metrics(c_weights)
q_ret, q_vol, q_sharpe = portfolio_metrics(q_weights)

# -------------------------------------------------
# TABS
# -------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Allocation",
    "📈 Efficient Frontier",
    "📉 Probability Distribution",
    "📄 PDF Report"
])

# -------------------------------------------------
# TAB 1 – Allocation Comparison
# -------------------------------------------------

with tab1:
    st.subheader("Portfolio Allocation Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tickers,
        y=c_weights,
        name="Classical",
    ))

    fig.add_trace(go.Bar(
        x=tickers,
        y=q_weights,
        name="Quantum-Inspired",
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Classical Sharpe Ratio", round(c_sharpe, 4))
    st.metric("Quantum Sharpe Ratio", round(q_sharpe, 4))

# -------------------------------------------------
# TAB 2 – Efficient Frontier (Animated)
# -------------------------------------------------

with tab2:
    st.subheader("Efficient Frontier Simulation")

    frontier_returns = []
    frontier_vol = []

    for i in range(3000):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        frontier_returns.append(ret)
        frontier_vol.append(vol)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=frontier_vol,
        y=frontier_returns,
        mode="markers",
        marker=dict(
            size=5,
            color=frontier_returns,
            colorscale="Viridis",
            showscale=True
        ),
        name="Simulated Portfolios"
    ))

    fig2.add_trace(go.Scatter(
        x=[c_vol],
        y=[c_ret],
        mode="markers",
        marker=dict(size=15),
        name="Classical"
    ))

    fig2.add_trace(go.Scatter(
        x=[q_vol],
        y=[q_ret],
        mode="markers",
        marker=dict(size=15),
        name="Quantum"
    ))

    fig2.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_title="Volatility",
        yaxis_title="Return"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# TAB 3 – Probability Distribution
# -------------------------------------------------

with tab3:
    st.subheader("Sharpe Ratio Distribution")

    fig3 = px.histogram(
        x=scores,
        nbins=50,
        template="plotly_dark",
        title="Quantum Optimization Sharpe Distribution"
    )

    fig3.add_vline(x=c_sharpe)
    fig3.add_vline(x=q_sharpe)

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# TAB 4 – PDF REPORT
# -------------------------------------------------

with tab4:
    st.subheader("Generate Professional Report")

    if st.button("Generate PDF Report"):

        filename = "Quantum_Portfolio_Report.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("<b>Quantum vs Classical Portfolio Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(f"Classical Sharpe Ratio: {round(c_sharpe,4)}", styles["Normal"]))
        elements.append(Paragraph(f"Quantum Sharpe Ratio: {round(q_sharpe,4)}", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Classical Weights:", styles["Heading2"]))
        for t, w in zip(tickers, c_weights):
            elements.append(Paragraph(f"{t}: {round(w,4)}", styles["Normal"]))

        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Quantum Weights:", styles["Heading2"]))
        for t, w in zip(tickers, q_weights):
            elements.append(Paragraph(f"{t}: {round(w,4)}", styles["Normal"]))

        doc.build(elements)

        with open(filename, "rb") as f:
            st.download_button("Download Report", f, file_name=filename)

st.markdown("---")
st.success("⚛️ Quantum Optimization Engine Running Successfully")
