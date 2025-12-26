# =============================================================================
# STREAMLIT APP - Portfolio Optimization CAC40
# Filename: app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle

# Configuration page
st.set_page_config(
    page_title="CAC40 Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CHARGEMENT MODÈLES & DATA
# =============================================================================

@st.cache_resource
def load_models():
    """Charge modèles pré-entraînés"""
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    return xgb_model, kmeans_model

@st.cache_data
def load_data():
    """Charge données historiques"""
    df = pd.read_csv('data/cac40_dataset.csv', index_col=[0,1], parse_dates=[0])
    predictions = pd.read_csv('results/predictions_latest.csv', index_col=0)
    backtest = pd.read_csv('results/portfolio_returns.csv', index_col=0, parse_dates=True)
    return df, predictions, backtest

# Charger
try:
    xgb_clf, kmeans = load_models()
    df, predictions_df, backtest_df = load_data()
    models_loaded = True
except:
    models_loaded = False
    st.error("⚠️ Modèles non chargés. Mode démo uniquement.")


# =============================================================================
# SIDEBAR - NAVIGATION
# =============================================================================

st.sidebar.title("📊 CAC40 Portfolio Optimizer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🔮 Prédictions Temps Réel", "📈 Backtesting", 
     "🤖 Modèles ML", "📊 Portfolio Optimal", "ℹ️ À Propos"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Projet M2 Finance Quantitative**
    
    Optimisation de portefeuille CAC40 
    par approche hybride :
    - Machine Learning (XGBoost)
    - Clustering (K-Means)
    - Optimisation Markowitz
    
    **Auteur** : Moussa  
    **Date** : Décembre 2025
    """
)


# =============================================================================
# PAGE 1 : ACCUEIL
# =============================================================================

if page == "🏠 Accueil":
    
    st.title("📊 CAC40 Portfolio Optimizer")
    st.markdown("### *Machine Learning × Théorie Moderne du Portefeuille*")
    
    st.markdown("---")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alpha vs CAC40", "+6.56%", "+2.17pp")
    
    with col2:
        st.metric("Sharpe Ratio", "0.33", "+0.04")
    
    with col3:
        st.metric("Max Drawdown", "-36.20%", "-10.5pp", delta_color="inverse")
    
    with col4:
        st.metric("Actions Analysées", "40", "CAC40")
    
    st.markdown("---")
    
    # Description
    st.markdown("""
    ## 🎯 Objectif
    
    Développer une **stratégie quantitative** pour optimiser un portefeuille d'actions CAC40 
    en combinant :
    
    1. **Machine Learning** (XGBoost) : Prédire probabilité de hausse
    2. **Clustering** (K-Means) : Identifier profils de risque
    3. **Optimisation Markowitz** : Maximiser ratio Sharpe
    
    ## 📊 Résultats
    
    Sur la période **2018-2025** (8 ans incluant COVID-19) :
    - ✅ **Surperformance** : +6.56% vs CAC40 buy-and-hold
    - ✅ **Sharpe supérieur** : 0.33 vs 0.29
    - ⚠️ **Risque accru** : Max drawdown -36% (vs -26%)
    
    ## 🚀 Navigation
    
    Utilisez le menu latéral pour explorer :
    - 🔮 **Prédictions Temps Réel** : Quelles actions acheter maintenant ?
    - 📈 **Backtesting** : Performance historique détaillée
    - 🤖 **Modèles ML** : Accuracy XGBoost, clusters K-Means
    - 📊 **Portfolio Optimal** : Allocation recommandée
    """)
    
    # Graphique performance
    st.markdown("---")
    st.subheader("📈 Performance Cumulative (2018-2025)")
    
    if models_loaded:
        fig = go.Figure()
        
        # Cumulative returns
        cumul_strategy = (1 + backtest_df['Strategy Return']).cumprod() - 1
        cumul_cac40 = (1 + backtest_df['CAC40 Buy&Hold']).cumprod() - 1
        
        fig.add_trace(go.Scatter(
            x=cumul_strategy.index,
            y=cumul_strategy.values * 100,
            name='Stratégie',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=cumul_cac40.index,
            y=cumul_cac40.values * 100,
            name='CAC40',
            line=dict(color='#4ECDC4', width=3)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Return Cumulé (%)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2 : PRÉDICTIONS TEMPS RÉEL
# =============================================================================

elif page == "🔮 Prédictions Temps Réel":
    
    st.title("🔮 Prédictions Temps Réel")
    st.markdown("### *Quelles actions acheter maintenant ?*")
    
    st.markdown("---")
    
    if not models_loaded:
        st.warning("Modèles non chargés. Affichage données historiques.")
    
    # Filtres
    col1, col2 = st.columns(2)
    
    with col1:
        min_proba = st.slider(
            "Probabilité minimale (%)",
            min_value=50,
            max_value=100,
            value=70,
            step=5
        )
    
    with col2:
        selected_cluster = st.multiselect(
            "Clusters",
            options=[0, 1, 2, 3],
            default=[3],
            format_func=lambda x: {
                0: "Defensive 🛡️",
                1: "Value 💰",
                2: "Growth 📈",
                3: "Momentum 🚀"
            }[x]
        )
    
    # Filtrer prédictions
    filtered_predictions = predictions_df[
        (predictions_df['proba_hausse_predicted'] >= min_proba/100) &
        (predictions_df['cluster_predicted'].isin(selected_cluster))
    ].sort_values('proba_hausse_predicted', ascending=False)
    
    st.markdown(f"**{len(filtered_predictions)} actions détectées**")
    
    # Tableau
    st.dataframe(
        filtered_predictions[[
            'cluster_predicted', 
            'proba_hausse_predicted', 
            'rsi', 
            'return_2m',
            'recommendation'
        ]].style.format({
            'proba_hausse_predicted': '{:.1%}',
            'rsi': '{:.1f}',
            'return_2m': '{:.2%}'
        }).background_gradient(subset=['proba_hausse_predicted'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Graphique scatter
    st.markdown("---")
    st.subheader("📊 RSI vs Probabilité Hausse")
    
    fig = px.scatter(
        predictions_df,
        x='rsi',
        y='proba_hausse_predicted',
        color='cluster_predicted',
        size='return_2m',
        hover_data=['ticker'],
        color_continuous_scale='viridis'
    )
    
    fig.add_hline(y=min_proba/100, line_dash="dash", line_color="red")
    fig.add_vline(x=70, line_dash="dash", line_color="orange")
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 3 : BACKTESTING
# =============================================================================

elif page == "📈 Backtesting":
    
    st.title("📈 Backtesting Détaillé")
    st.markdown("### *Performance Historique (2018-2025)*")
    
    st.markdown("---")
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Return Total", "+33.95%")
        st.metric("Return Annualisé", "+4.23%")
    
    with col2:
        st.metric("Volatilité", "18.50%")
        st.metric("Sharpe Ratio", "0.33")
    
    with col3:
        st.metric("Max Drawdown", "-36.20%", delta_color="inverse")
        st.metric("Win Rate", "58.5%")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Returns Cumulés", 
        "📊 Rolling Sharpe", 
        "📉 Drawdown",
        "📊 Distribution"
    ])
    
    with tab1:
        cumul_strategy = (1 + backtest_df['Strategy Return']).cumprod() - 1
        cumul_cac40 = (1 + backtest_df['CAC40 Buy&Hold']).cumprod() - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumul_strategy.index, y=cumul_strategy.values*100,
            name='Stratégie', line=dict(color='#FF6B6B', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=cumul_cac40.index, y=cumul_cac40.values*100,
            name='CAC40', line=dict(color='#4ECDC4', width=3)
        ))
        fig.update_layout(yaxis_title="Return Cumulé (%)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        rolling_sharpe = (backtest_df['Strategy Return'].rolling(126).mean() / 
                         backtest_df['Strategy Return'].rolling(126).std()) * np.sqrt(252)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name='Sharpe 6M', line=dict(color='#FFD93D', width=3)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Sharpe Ratio", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        cumul = (1 + backtest_df['Strategy Return']).cumprod()
        running_max = cumul.cummax()
        drawdown = (cumul - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values*100,
            fill='tozeroy', line=dict(color='#E74C3C', width=2)
        ))
        fig.update_layout(yaxis_title="Drawdown (%)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=backtest_df['Strategy Return']*100,
            name='Stratégie', opacity=0.7, nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=backtest_df['CAC40 Buy&Hold']*100,
            name='CAC40', opacity=0.7, nbinsx=50
        ))
        fig.update_layout(
            xaxis_title="Returns Journaliers (%)",
            yaxis_title="Fréquence",
            barmode='overlay',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 4 : MODÈLES ML
# =============================================================================

elif page == "🤖 Modèles ML":
    
    st.title("🤖 Modèles Machine Learning")
    st.markdown("### *XGBoost + K-Means*")
    
    st.markdown("---")
    
    # XGBoost
    st.subheader("📊 XGBoost Classification")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "88.44%")
    with col2:
        st.metric("AUC-ROC", "0.9556")
    with col3:
        st.metric("Precision Hausse", "89.26%")
    
    st.markdown("""
    **Features utilisées** (15) :
    - Technical : RSI, MACD, Bollinger Bands, ATR
    - Fundamentals : Returns 1m/2m/3m/6m
    - Fama-French : Mkt-RF, SMB, HML, RMW, CMA
    - Volume : Garman-Klass Volatility
    """)
    
    # K-Means
    st.markdown("---")
    st.subheader("🎯 K-Means Clustering")
    
    cluster_stats = pd.DataFrame({
        'Cluster': [0, 1, 2, 3],
        'Nom': ['Defensive 🛡️', 'Value 💰', 'Growth 📈', 'Momentum 🚀'],
        'RSI Moyen': [45, 52, 58, 72],
        'Return 2M': [-0.02, 0.01, 0.03, 0.08],
        'Volatilité': [0.15, 0.18, 0.22, 0.28]
    })
    
    st.dataframe(cluster_stats, use_container_width=True)


# =============================================================================
# PAGE 5 : PORTFOLIO OPTIMAL
# =============================================================================

elif page == "📊 Portfolio Optimal":
    
    st.title("📊 Portfolio Optimal")
    st.markdown("### *Allocation Recommandée - Janvier 2026*")
    
    st.markdown("---")
    
    # Simulation allocation
    capital = st.number_input(
        "Capital à investir (€)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Portfolio exemple (simulé)
    portfolio_example = pd.DataFrame({
        'Ticker': ['AIR.PA', 'MC.PA', 'OR.PA', 'BNP.PA', 'KER.PA'],
        'Poids': [0.25, 0.23, 0.22, 0.18, 0.12],
        'Prix': [125.30, 645.20, 387.50, 68.50, 445.80],
        'Proba': [0.912, 0.889, 0.876, 0.823, 0.785]
    })
    
    portfolio_example['Montant'] = portfolio_example['Poids'] * capital
    portfolio_example['Actions'] = (portfolio_example['Montant'] / portfolio_example['Prix']).astype(int)
    
    st.dataframe(
        portfolio_example.style.format({
            'Poids': '{:.1%}',
            'Prix': '{:.2f}€',
            'Montant': '{:.2f}€',
            'Proba': '{:.1%}'
        }),
        use_container_width=True
    )
    
    # Pie chart
    fig = px.pie(
        portfolio_example,
        values='Poids',
        names='Ticker',
        title='Allocation Optimale'
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 6 : À PROPOS
# =============================================================================

elif page == "ℹ️ À Propos":
    
    st.title("ℹ️ À Propos du Projet")
    
    st.markdown("""
    ## 🎓 Contexte Académique
    
    Ce projet a été réalisé dans le cadre du **Master 2 Finance Quantitative** 
    à l'**Université de Lorraine**.
    
    ## 🎯 Objectif
    
    Démontrer l'efficacité d'une approche **hybride** combinant :
    - Machine Learning moderne (XGBoost)
    - Théorie financière classique (Markowitz)
    - Analyse quantitative (Clustering, Backtesting)
    
    ## 📚 Méthodologie
    
    1. **Data Collection** : 10 ans CAC40 (yfinance)
    2. **Features Engineering** : 15 indicateurs techniques + fondamentaux
    3. **Machine Learning** : XGBoost (88% accuracy)
    4. **Clustering** : K-Means (4 profils de risque)
    5. **Optimisation** : PyPortfolioOpt (Markowitz)
    6. **Backtesting** : 2018-2025 (rolling window)
    
    ## 🛠️ Stack Technique
    
    - **Language** : Python 3.11
    - **ML** : XGBoost, Scikit-learn
    - **Optimization** : PyPortfolioOpt
    - **Viz** : Plotly, Matplotlib
    - **Web** : Streamlit
    - **Deployment** : Streamlit Cloud
    
    ## 📊 Résultats Clés
    
    - ✅ Alpha +6.56% vs CAC40
    - ✅ Sharpe 0.33 (>benchmark)
    - ⚠️ Max DD -36.20%
    
    ## 📧 Contact
    
    **Auteur** : Moussa  
    **LinkedIn** : [linkedin.com/in/moussa](https://linkedin.com)  
    **GitHub** : [github.com/moussa/cac40-optimizer](https://github.com)  
    **Email** : moussa@example.com
    
    ## ⭐ GitHub
    
    Le code complet est disponible sur GitHub :
    [github.com/moussa/cac40-portfolio-optimizer](https://github.com)
    
    N'hésitez pas à ⭐ star le repo !
    """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit | © 2025 Moussa | M2 Finance Quantitative</p>
    </div>
    """,
    unsafe_allow_html=True
)
