# web_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import tempfile
import os

# Import the analyzer directly
from anomaly_detector import ProcurementAnomalyDetector

st.set_page_config(
    page_title="Procurement Risk Dashboard",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def analyze_uploaded_file(uploaded_file):
    """Analyze uploaded procurement data"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize detector and analyze
        detector = ProcurementAnomalyDetector()
        df = detector.load_and_validate(tmp_file_path)
        results = detector.calculate_risk_scores(df)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return results, df
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise e

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

def generate_text_report(results, df):
    """Generate text summary report"""
    total = len(results)
    high_risk = len(results[results['risk_level'] == 'RED'])
    medium_risk = len(results[results['risk_level'] == 'YELLOW'])
    low_risk = len(results[results['risk_level'] == 'GREEN'])
    avg_risk = results['risk_score'].mean()
    
    report = f"""PROCUREMENT RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
==================
Total Contracts Analyzed: {total:,}
Date Range: {df['accept_date'].min()} to {df['accept_date'].max()}

RISK DISTRIBUTION
=================
High Risk (RED): {high_risk:,} ({high_risk/total*100:.1f}%)
Medium Risk (YELLOW): {medium_risk:,} ({medium_risk/total*100:.1f}%)
Low Risk (GREEN): {low_risk:,} ({low_risk/total*100:.1f}%)

Average Risk Score: {avg_risk:.3f}

RISK FACTORS (Average Scores)
=============================
Price Anomaly: {results['price_anomaly'].mean():.3f}
Supplier Concentration: {results['supplier_concentration'].mean():.3f}
Contract Splitting: {results['contract_splitting'].mean():.3f}
Low Competition: {results['low_competition'].mean():.3f}

HIGH-RISK ANALYSIS
==================
Total High-Risk Contracts: {high_risk:,}
Total Value at Risk: {results[results['risk_level'] == 'RED']['contract_sum'].sum():,.2f} KZT

TOP 10 HIGHEST RISK CONTRACTS
=============================
"""
    
    top_10 = results.nlargest(10, 'risk_score')
    for idx, row in top_10.iterrows():
        report += f"\nLot {row['lot_id']}: Risk={row['risk_score']:.3f} - {row['contract_sum']:,.0f} KZT"
    
    return report

def main():
    st.title("🔍 Procurement Risk Analysis Dashboard")
    st.markdown("AI-powered anomaly detection for government procurement")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Input")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload Procurement Data (CSV)",
            type=['csv'],
            help="Upload your procurement data CSV file (e.g., OrderDetail.csv)"
        )
        
        # Analysis settings
        if uploaded_file:
            st.success(f"✓ Loaded: {uploaded_file.name}")
            
            st.header("⚙️ Analysis Settings")
            
            # Risk threshold adjustments
            with st.expander("Adjust Risk Thresholds"):
                price_weight = st.slider("Price Anomaly Weight", 0.0, 1.0, 0.25, 0.05)
                concentration_weight = st.slider("Concentration Weight", 0.0, 1.0, 0.30, 0.05)
                splitting_weight = st.slider("Splitting Weight", 0.0, 1.0, 0.20, 0.05)
                competition_weight = st.slider("Competition Weight", 0.0, 1.0, 0.25, 0.05)
                
                # Normalize weights
                total_weight = price_weight + concentration_weight + splitting_weight + competition_weight
                if total_weight > 0:
                    weights = {
                        'price': price_weight / total_weight,
                        'concentration': concentration_weight / total_weight,
                        'splitting': splitting_weight / total_weight,
                        'competition': competition_weight / total_weight
                    }
                    st.info(f"Weights normalized to sum to 1.0")
    
    # Main content area
    if uploaded_file:
        try:
            # Analyze data
            with st.spinner("🔄 Analyzing procurement data..."):
                results, original_df = analyze_uploaded_file(uploaded_file)
            
            st.success("✅ Analysis complete!")
            
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
                # Generate summary report
                summary_report = generate_text_report(results, original_df)
                
                st.download_button(
                    label="📄 Download Summary Report",
                    data=summary_report,
                    file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Error analyzing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format with columns: lot_id, provider_bin, customer_bin, contract_sum, paid_sum, accept_date, order_method_id")
    
    else:
        # Landing page when no file is uploaded
        st.info("👈 Please upload a procurement data CSV file to begin analysis")
        
        # Instructions and information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 Required CSV Format
            
            Your CSV file should contain these columns:
            - `lot_id` - Lot identifier
            - `provider_bin` - Provider BIN/ID
            - `customer_bin` - Customer BIN/ID
            - `contract_sum` - Contract value
            - `paid_sum` - Amount paid
            - `accept_date` - Contract date
            - `order_method_id` - Procurement method ID
            
            ### 📁 File Requirements
            - CSV format
            - UTF-8 encoding recommended
            - Date format: YYYY-MM-DD or DD.MM.YYYY
            """)
        
        with col2:
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
        
        # Sample data generator
        st.markdown("---")
        st.subheader("🧪 Generate Sample Data")
        
        if st.button("Generate Sample CSV"):
            sample_data = pd.DataFrame({
                'lot_id': range(1, 101),
                'provider_bin': np.random.choice(['1234567890', '0987654321', '1111111111', '2222222222'], 100),
                'customer_bin': np.random.choice(['9999999999', '8888888888', '7777777777'], 100),
                'contract_sum': np.random.exponential(100000, 100),
                'paid_sum': np.random.exponential(100000, 100),
                'accept_date': pd.date_range('2023-01-01', periods=100, freq='D'),
                'order_method_id': np.random.choice([1, 2, 3, 4, 5, 6], 100)
            })
            
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_procurement_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()