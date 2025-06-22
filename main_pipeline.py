
import sys
import logging
from anomaly_detector import ProcurementAnomalyDetector
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_analysis(input_file: str, output_file: str = 'risk_analysis_results.csv'):
    """Main pipeline execution"""
    logger.info(f"Starting procurement risk analysis for: {input_file}")
    
    # Initialize detector
    detector = ProcurementAnomalyDetector()
    
    # Load and validate data
    logger.info("Loading and validating data...")
    df = detector.load_and_validate(input_file)
    logger.info(f"Loaded {len(df)} valid contracts")
    
    # Run risk analysis
    logger.info("Running risk analysis...")
    results = detector.calculate_risk_scores(df)
    
    # Save results
    results.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")
    
    # Generate summary statistics
    summary = {
        'total_contracts': len(results),
        'high_risk': len(results[results['risk_level'] == 'RED']),
        'medium_risk': len(results[results['risk_level'] == 'YELLOW']),
        'low_risk': len(results[results['risk_level'] == 'GREEN']),
        'avg_risk_score': results['risk_score'].mean()
    }
    
    logger.info("\n=== RISK ANALYSIS SUMMARY ===")
    logger.info(f"Total contracts analyzed: {summary['total_contracts']:,}")
    logger.info(f"High risk (RED): {summary['high_risk']:,} ({summary['high_risk']/summary['total_contracts']*100:.1f}%)")
    logger.info(f"Medium risk (YELLOW): {summary['medium_risk']:,} ({summary['medium_risk']/summary['total_contracts']*100:.1f}%)")
    logger.info(f"Low risk (GREEN): {summary['low_risk']:,} ({summary['low_risk']/summary['total_contracts']*100:.1f}%)")
    logger.info(f"Average risk score: {summary['avg_risk_score']:.3f}")
    
    # Top risks
    logger.info("\n=== TOP 10 HIGHEST RISK CONTRACTS ===")
    top_risks = results.nlargest(10, 'risk_score')[
        ['lot_id', 'provider_bin', 'customer_bin', 'contract_sum', 'risk_score', 'risk_level']
    ]
    for idx, row in top_risks.iterrows():
        logger.info(f"Lot {row['lot_id']}: Risk={row['risk_score']:.3f} ({row['risk_level']}) - {row['contract_sum']:,.0f} KZT")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <input_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    run_analysis(input_file)