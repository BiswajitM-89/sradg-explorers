import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
class FinancialAnomalyDetector:
    def __init__(self, historical_data_path):
        self.historical_data = self.load_data(historical_data_path)
        self.preprocessor = None
        self.model = None
        self._prepare_system()
    def load_data(self, path):
            """Load financial data from various formats"""
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                raise ValueError("Unsupported file format")
            
            # Standardize datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        
    def _prepare_system(self):
        """Build detection pipeline with automated feature engineering"""
        # Feature engineering
        self.historical_data['day_of_week'] = self.historical_data['timestamp'].dt.dayofweek
        self.historical_data['is_month_end'] = self.historical_data['timestamp'].dt.is_month
        
        # Create training features
        features = ['amount', 'day_of_week', 'is_month_end']
        X = self.historical_data[features]
        
        # Build processing pipeline
        self.preprocessor = Pipeline([
            ('scaler', StandardScaler())
	])

        # Train anomaly detection model
        self.model = IsolationForest(
            n_estimators=100,
	    contamination=0.01,
	    random_state=42
        )

        X_preprocessed = self.preprocessor.fit_transform(X)
        self.model.fit(X_preprocessed)
        
    def detect_anomalies(self, real_time_data):
        """Detect and classify anomalies in real-time data"""
        # Feature engineering for new data
        real_time_data = real_time_data.copy()
        real_time_data['day_of_week'] = real_time_data['timestamp'].dt.dayofweek
        real_time_data['is_month_end'] = real_time_data['timestamp'].dt.is_month_end.astype(int)

        # Preprocess and predict
        features = ['amount', 'day_of_week', 'is_month_end']
        X_new = self.preprocessor.transform(real_time_data[features])
        anomalies = self.model.predict(X_new)

        # Add anomaly flags and scores
        real_time_data['anomaly_score'] = self.model.decision_function(X_new)
        real_time_data['is_anomaly']= np.where(anomalies == -1, True, False)
	
        # Classify anomaly types
        real_time_data['anomaly_type'] = real_time_data.apply(
        self._classify_anomaly, axis=1
	)
        return real_time_data
    
    def _classify_anomaly(self, row):
            """Rule based anomaly classification system"""
            if not row['is_anomaly']:
                return 'Normal'
            
            historical_avg = self.historical_data['amount'].mean()
            historical_std = self.historical_data['amount'].std()
	
        # Amount-based anomalies
            if row['amount'] > historical_avg + 3*historical_std:
                return 'High Value Transaction'
            elif row['amount'] < historical_avg - 3*historical_std:
                return 'Low Value Outlier'

        # Temporal anomalies
            time_diff = (row['timestamp'] - self.historical_data['timestamp'].max()).days
            if abs(time_diff) > 7:
                return 'Unusual Timing Pattern'

        # Contextual anomalies
            if row['day_of_week'] in [5,6] and row['amount'] > historical_avg:
                return 'Unusual Weekend Activity'
                return 'Unclassified Anomaly'
    
class ReconciliationAI:
    def __init__(self):self.anomaly_context = {
                'High Value Transactions': "Verify merchant approval and AM Checks",
                'Low Value Outlier': "Check for system rounding errors or partial failures",
                'Unusual Timing Pattern': "Review batch processing schedules and timezone set",
                'Unusual Weekend Activity': "Validate holiday/weekend business operations"
            }
	
    def analyze_breaks(self, anomalies_df):
        'AI Powered reconciliation break analysis'
        #Agentic analysis (can integrate with LLM API)
        summary = []
            
        for _, row in anomalies_df[anomalies_df['is_anomaly']].iterrows():
                analysis = f"""
                **Anomaly Detected** ({row['timestamp']}):
                        - Type: {row['anomaly_type']}
                        - Amount: {row['amount']}
                        - Recommended Action: {self.anomaly_context.get(row['anomaly_type'],
							'Review transaction manually')}
							"""
                summary.append(analysis)
                
                return "\\n".join(summary)
        
            # Example Usage
        if __name__ == "__main__":
                #Initialize system with historical data
                detector = FinancialAnomalyDetector('financial_transactions_12m.csv')


                # Simulate real-time data
                real_time_data = pd.DataFrame({
                'timestamp': [datetime.now() - timedelta(hours=x) for x in range(24)],
		'amount': np.concatenate([
                    np.random.normal(1000, 200, 23),
                    np.array([5000]) # Inject anomaly
		])
            })

            # Detect anomalies
                results = detector.detect_anomalies(real_time_data)
            
            # Generate reconciliation report
                ai_agent = ReconciliationAI()
                report = ai_agent.analyze_breaks(results)
            
                print("Anomaly Detection Results:")
                print(results[results['is_anomaly']])
                print("\\nAI Reconciliation Summary:")
                print(report)
