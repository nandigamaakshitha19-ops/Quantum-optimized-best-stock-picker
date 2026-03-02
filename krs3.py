# =====================================================
# ELITE QUANTUM PORTFOLIO OPTIMIZATION PLATFORM
# Animated Institutional Dashboard Version
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(page_title="Elite Quantum Portfolio Engine",
                   layout="wide")

st.title("⚛️ Elite Quantum Portfolio Optimization Engine")
st.markdown("### CVaR-QAOA | Budget-Constrained | Animated Institutional Dashboard")

# ==========================
# COMPANY DATA
# ==========================

stocks = ["JPMorgan Chase", "BlackRock", "Fidelity", "Robinhood"]
n = 4

returns = np.array([12.5, 10.2, 9.8, 15.3])
prices = np.array([18, 22, 16, 14])

# ==========================
# CONFIGURATION PANEL
# ==========================

with st.expander("⚙️ Configuration", expanded=True):
    B = st.slider("💰 Budget", 20, 80, 45)
    risk_weight = st.slider("Risk Sensitivity", 0.5, 5.0, 2.0)
    p_layers = st.selectbox("QAOA Layers", [1,2,3], index=1)
    shots = st.selectbox("Shots", [2048,4096,8192], index=1)
    cvar_alpha = st.slider("CVaR Alpha", 0.1,0.5,0.3)

# ==========================
# MARKET SIMULATION
# ==========================

np.random.seed(42)
mean = np.random.uniform(0.001,0.02,n)
base_cov = np.random.uniform(0.0001,0.002,(n,n))
base_cov = (base_cov + base_cov.T)/2
cov_matrix = base_cov @ base_cov.T
sim_returns = np.random.multivariate_normal(mean, cov_matrix, 800)
cov_matrix = np.cov(sim_returns.T)

# ==========================
# CLASSICAL SOLVER (Exact)
# ==========================

def classical_solver():
    best_val=-1e9
    best=None
    for bits in product([0,1],repeat=n):
        x=np.array(bits)
        if np.dot(prices,x)<=B:
            val=np.dot(returns,x)-risk_weight*(x.T@cov_matrix@x)
            if val>best_val:
                best_val=val
                best=x
    return best,best_val

# ==========================
# QAOA CIRCUIT
# ==========================

def qaoa_circuit(params):
    qc=QuantumCircuit(n)
    qc.h(range(n))
    gammas=params[:p_layers]
    betas=params[p_layers:]
    for layer in range(p_layers):
        for i in range(n):
            qc.rz(-gammas[layer]*returns[i],i)
        for i in range(n):
            qc.rx(2*betas[layer],i)
    qc.measure_all()
    return qc

simulator=AerSimulator()

# ==========================
# RUN BUTTON
# ==========================

if st.button("🚀 Run Animated Optimization"):

    classical_bits,classical_val=classical_solver()

    energy_history=[]
    progress_chart = st.empty()

    best_quantum_val=-1e9
    best_state=None
    final_counts=None

    # ======================
    # ANIMATED OPTIMIZATION
    # ======================

    for restart in range(3):

        def cvar_expectation(params):
            qc=qaoa_circuit(params)
            result=simulator.run(qc,shots=shots).result()
            counts=result.get_counts()
            energies=[]
            for bitstring,count in counts.items():
                x=np.array([int(b) for b in bitstring[::-1]])
                ret=np.dot(returns,x)
                risk=x.T@cov_matrix@x
                energy=-(ret-risk_weight*risk)
                energies+=[energy]*count
            energies=np.array(energies)
            cutoff=int(len(energies)*cvar_alpha)
            val=np.mean(np.sort(energies)[:cutoff])
            energy_history.append(val)

            # Live animation
            fig_anim=go.Figure()
            fig_anim.add_trace(go.Scatter(
                y=pd.Series(energy_history).rolling(3,min_periods=1).mean(),
                mode='lines'
            ))
            fig_anim.update_layout(
                template="plotly_dark",
                title="Live QAOA Energy Optimization",
                xaxis_title="Iteration",
                yaxis_title="Energy"
            )
            progress_chart.plotly_chart(fig_anim,use_container_width=True)

            return val

        init=np.random.rand(2*p_layers)
        result=minimize(cvar_expectation,init,
                        method="COBYLA",
                        options={'maxiter':80})

        qc_final=qaoa_circuit(result.x)
        res=simulator.run(qc_final,shots=shots).result()
        counts=res.get_counts()
        state=max(counts,key=counts.get)
        x=np.array([int(b) for b in state[::-1]])
        val=np.dot(returns,x)-risk_weight*(x.T@cov_matrix@x)

        if val>best_quantum_val:
            best_quantum_val=val
            best_state=state
            final_counts=counts

    x=np.array([int(b) for b in best_state[::-1]])

    portfolio_return=np.dot(returns,x)
    portfolio_risk=x.T@cov_matrix@x
    portfolio_cost=np.dot(prices,x)

    sharpe=(portfolio_return-0.02)/(np.sqrt(portfolio_risk)+1e-6)

    # ======================
    # DASHBOARD METRICS
    # ======================

    st.markdown("## 📊 Portfolio KPIs")

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Investment",round(portfolio_cost,2))
    c2.metric("Return",round(portfolio_return,2))
    c3.metric("Risk",round(portfolio_risk,4))
    c4.metric("Sharpe Ratio",round(sharpe,3))
    c5.metric("Quantum Score",round(best_quantum_val,3))

    # ======================
    # ALLOCATION PIE
    # ======================

    fig_pie=px.pie(
        names=stocks,
        values=x,
        title="Portfolio Allocation",
        template="plotly_dark"
    )
    st.plotly_chart(fig_pie,use_container_width=True)

    # ======================
    # PROBABILITY DISTRIBUTION
    # ======================

    total=sum(final_counts.values())
    probs={k:v/total for k,v in final_counts.items()}
    sorted_probs=dict(sorted(probs.items(),
                             key=lambda x:x[1],
                             reverse=True))

    states=list(sorted_probs.keys())[:8]
    values=list(sorted_probs.values())[:8]

    fig_prob=go.Figure(go.Bar(x=states,y=values))
    fig_prob.update_layout(
        template="plotly_dark",
        title="Quantum Measurement Distribution"
    )
    st.plotly_chart(fig_prob,use_container_width=True)

    # ======================
    # EFFICIENT FRONTIER
    # ======================

    rand_returns=[]
    rand_risks=[]

    for _ in range(600):
        rb=np.random.randint(0,2,n)
        if np.dot(prices,rb)<=B:
            rand_returns.append(np.dot(returns,rb))
            rand_risks.append(rb.T@cov_matrix@rb)

    fig_frontier=go.Figure()

    fig_frontier.add_trace(go.Scatter(
        x=rand_risks,
        y=rand_returns,
        mode='markers',
        marker=dict(
            size=6,
            color=rand_returns,
            colorscale="Viridis",
            showscale=True
        ),
        name="Feasible Portfolios"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[portfolio_risk],
        y=[portfolio_return],
        mode='markers',
        marker=dict(size=18,symbol="star",color="red"),
        name="Quantum Optimum"
    ))

    fig_frontier.update_layout(
        template="plotly_dark",
        title="Efficient Frontier (Interactive)"
    )

    st.plotly_chart(fig_frontier,use_container_width=True)

    # ======================
    # RETURN DISTRIBUTION
    # ======================

    portfolio_sim=sim_returns@x
    VaR=np.percentile(portfolio_sim,5)
    CVaR=np.mean(portfolio_sim[portfolio_sim<=VaR])

    fig_hist=px.histogram(
        portfolio_sim,
        nbins=50,
        template="plotly_dark",
        title="Monte Carlo Return Distribution"
    )
    fig_hist.add_vline(x=VaR,line_color="red")
    st.plotly_chart(fig_hist,use_container_width=True)

    st.metric("Value at Risk (5%)",round(VaR,4))
    st.metric("Conditional VaR",round(CVaR,4))

    # ======================
    # COVARIANCE HEATMAP
    # ======================

    fig_heat=px.imshow(
        cov_matrix,
        x=stocks,
        y=stocks,
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Covariance Matrix"
    )
    st.plotly_chart(fig_heat,use_container_width=True)

    # ======================
    # PDF REPORT
    # ======================

    file_path="Quantum_Portfolio_Report.pdf"
    doc=SimpleDocTemplate(file_path)
    elements=[]
    styles=getSampleStyleSheet()

    elements.append(Paragraph("Elite Quantum Portfolio Report",styles['Heading1']))
    elements.append(Spacer(1,12))
    elements.append(Paragraph(f"Quantum Score: {best_quantum_val}",styles['Normal']))
    elements.append(Spacer(1,12))

    table_data=[["Company","Chosen","Return","Cost"]]
    for i in range(n):
        table_data.append([stocks[i],int(x[i]),returns[i],prices[i]])

    table=Table(table_data)
    table.setStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ])

    elements.append(table)
    doc.build(elements)

    with open(file_path,"rb") as f:
        st.download_button("📥 Download PDF Report",
                           f,
                           file_name="Quantum_Portfolio_Report.pdf")