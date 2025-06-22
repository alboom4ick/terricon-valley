# web_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from anomaly_detector import ProcurementAnomalyDetector
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="Procurement Risk Dashboard",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_and_analyze(uploaded_file):
    """Load and analyze procurement data"""
    detector = ProcurementAnomalyDetector()
    df = detector.load_and_validate(uploaded_file)
    results = detector.calculate_risk_scores(df)
    return results

def create_risk_gauge(risk_score):
    """Create risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Risk Score"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.65], 'color': "yellow"},
                {'range': [0.65, 1], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.65
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.title("🔍 Procurement Risk Analysis Dashboard")
    st.markdown("AI-powered anomaly detection for government procurement")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload OrderDetail.csv file"
        )
        
        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
    
    if uploaded_file:
        # Analyze data
        with st.spinner("Analyzing procurement data..."):
            results = load_and_analyze(uploaded_file)
        
        # Overview metrics
        st.header("📊 Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total = len(results)
        high_risk = len(results[results['risk_level'] == 'RED'])
        medium_risk = len(results[results['risk_level'] == 'YELLOW'])
        low_risk = len(results[results['risk_level'] == 'GREEN'])
        avg_risk = results['risk_score'].mean()
        
        col1.metric("Total Contracts", f"{total:,}")
        col2.metric("High Risk", f"{high_risk:,}", f"{high_risk/total*100:.1f}%")
        col3.metric("Medium Risk", f"{medium_risk:,}", f"{medium_risk/total*100:.1f}%")
        col4.metric("Low Risk", f"{low_risk:,}", f"{low_risk/total*100:.1f}%")
        col5.metric("Avg Risk", f"{avg_risk:.3f}")
        
        # Risk distribution
        st.header("📈 Risk Analysis")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Risk level pie chart
            risk_counts = results['risk_level'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={
                    'RED': '#FF4444',
                    'YELLOW': '#FFB000',
                    'GREEN': '#00C851'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Risk score histogram
            fig_hist = px.histogram(
                results,
                x='risk_score',
                nbins=50,
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Contracts'}
            )
            fig_hist.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col3:
            # Overall risk gauge
            fig_gauge = create_risk_gauge(avg_risk)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk factors analysis
        st.header("🎯 Risk Factors Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average risk by factor
            risk_factors = {
                'Price Anomaly': results['price_anomaly'].mean(),
                'Supplier Concentration': results['supplier_concentration'].mean(),
                'Contract Splitting': results['contract_splitting'].mean(),
                'Low Competition': results['low_competition'].mean()
            }
            
            fig_factors = go.Figure(data=[
                go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    marker_color=['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
                )
            ])
            fig_factors.update_layout(
                title="Average Risk Score by Factor",
                xaxis_title="Risk Factor",
                yaxis_title="Average Score",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        
        with col2:
            # Risk factors correlation
            high_risk_contracts = results[results['risk_level'] == 'RED']
            if len(high_risk_contracts) > 0:
                risk_factor_cols = ['price_anomaly', 'supplier_concentration', 
                                   'contract_splitting', 'low_competition']
                
                # Calculate percentage of high-risk contracts with each factor > 0.5
                factor_prevalence = {}
                for factor in risk_factor_cols:
                    prevalence = (high_risk_contracts[factor] > 0.5).mean() * 100
                    factor_prevalence[factor.replace('_', ' ').title()] = prevalence
                
                fig_prevalence = go.Figure(data=[
                    go.Bar(
                        x=list(factor_prevalence.values()),
                        y=list(factor_prevalence.keys()),
                        orientation='h',
                        marker_color='#e74c3c'
                    )
                ])
                fig_prevalence.update_layout(
                    title="Risk Factor Prevalence in High-Risk Contracts",
                    xaxis_title="Percentage (%)",
                    yaxis_title="Risk Factor"
                )
                st.plotly_chart(fig_prevalence, use_container_width=True)
        
        # Time series analysis
        st.header("📅 Temporal Analysis")
        
        results['year_month'] = pd.to_datetime(results['accept_date']).dt.to_period('M')
        monthly_risk = results.groupby(['year_month', 'risk_level']).size().unstack(fill_value=0)
        
        fig_timeline = go.Figure()
        for risk_level, color in [('RED', '#FF4444'), ('YELLOW', '#FFB000'), ('GREEN', '#00C851')]:
            if risk_level in monthly_risk.columns:
                fig_timeline.add_trace(go.Scatter(
                    x=monthly_risk.index.astype(str),
                    y=monthly_risk[risk_level],
                    mode='lines+markers',
                    name=risk_level,
                    line=dict(color=color, width=2)
                ))
        
        fig_timeline.update_layout(
            title="Risk Levels Over Time",
            xaxis_title="Month",
            yaxis_title="Number of Contracts",
            hovermode='x unified'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # High risk contracts table
        st.header("🚨 High Risk Contracts")
        
        high_risk_df = results[results['risk_level'] == 'RED'].nlargest(50, 'risk_score')
        
        if len(high_risk_df) > 0:
            # Add risk indicators
            high_risk_df['risk_indicators'] = high_risk_df.apply(
                lambda row: ', '.join([
                    'Price' if row['price_anomaly'] > 0.5 else '',
                    'Concentration' if row['supplier_concentration'] > 0.5 else '',
                    'Splitting' if row['contract_splitting'] > 0.5 else '',
                    'Competition' if row['low_competition'] > 0.5 else ''
                ]).strip(', '),
                axis=1
            )
            
            display_columns = [
                'lot_id', 'provider_bin', 'customer_bin', 
                'contract_sum', 'risk_score', 'risk_indicators'
            ]
            
            st.dataframe(
                high_risk_df[display_columns].style.format({
                    'contract_sum': '{:,.0f}',
                    'risk_score': '{:.3f}'
                }).background_gradient(subset=['risk_score'], cmap='Reds'),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No high-risk contracts detected")
        
        # Download section
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Analysis (CSV)",
                data=csv,
                file_name=f"procurement_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            high_risk_csv = high_risk_df.to_csv(index=False)
            st.download_button(
                label="🚨 Download High Risk Contracts",
                data=high_risk_csv,
                file_name=f"high_risk_contracts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Summary report
            summary = f"""PROCUREMENT RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
==================
Total Contracts: {total:,}
High Risk: {high_risk:,} ({high_risk/total*100:.1f}%)
Medium Risk: {medium_risk:,} ({medium_risk/total*100:.1f}%)
Low Risk: {low_risk:,} ({low_risk/total*100:.1f}%)
Average Risk Score: {avg_risk:.3f}

RISK FACTORS (Average Scores)
=============================
Price Anomaly: {results['price_anomaly'].mean():.3f}
Supplier Concentration: {results['supplier_concentration'].mean():.3f}
Contract Splitting: {results['contract_splitting'].mean():.3f}
Low Competition: {results['low_competition'].mean():.3f}

TOP 10 HIGHEST RISK CONTRACTS
=============================
"""
            top_10 = results.nlargest(10, 'risk_score')
            for idx, row in top_10.iterrows():
                summary += f"\nLot {row['lot_id']}: Risk={row['risk_score']:.3f} - {row['contract_sum']:,.0f} KZT"
            
            st.download_button(
                label="📄 Download Summary Report",
                data=summary,
                file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        # Landing page
        st.info("👈 Please upload OrderDetail.csv file to begin analysis")
        
        st.markdown("""
        ### 🎯 What this tool detects:
        
        1. **Price Anomalies** - Contracts with suspiciously high prices
        2. **Supplier Concentration** - Frequent wins by same suppliers
        3. **Contract Splitting** - Artificial division of large contracts
        4. **Low Competition** - Single bidder patterns
        
        ### 📊 Risk Levels:
        
        - 🟢 **GREEN**: Low risk (0.00 - 0.30)
        - 🟡 **YELLOW**: Medium risk (0.30 - 0.65)
        - 🔴 **RED**: High risk (0.65 - 1.00)
        """)

if __name__ == "__main__":
    main()