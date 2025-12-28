import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load the saved model
model = joblib.load('fraud_model.pkl')

def check_transaction(data):
    # This function predicts if a transaction is fraud (1) or genuine (0)
    prediction = model.predict(data)
    if prediction[0] == 1:
        return "⚠️ ALERT: Fraudulent Transaction Detected!"
    else:
        return "✅ Transaction Verified: Genuine."

# 2. Prepare the test data (Scaling must match training!)
try:
    df_sample = pd.read_csv('creditcard.csv')
    
    # STEP 1: Scale the 'Amount' just like we did in training
    scaler = StandardScaler()
    df_sample['scaled_amount'] = scaler.fit_transform(df_sample['Amount'].values.reshape(-1, 1))
    
    # STEP 2: Drop the columns the model doesn't expect ('Time' and original 'Amount')
    # Keep 'Class' separate
    test_row = df_sample.drop(['Class', 'Time', 'Amount'], axis=1).iloc[0:1] 
    
    print("Testing System...")
    # Now the feature names will match: 'scaled_amount' will be there!
    print(check_transaction(test_row))
    
except Exception as e:
    print(f"Error: {e}")
