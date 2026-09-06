# Road Accident Severity Prediction - By Neeraj Kumar
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Road Accident Severity Prediction")

# Sample logic - aapka original data yaha ayega
data = {'speed':[30,50,80,60,90], 'weather':[1,0,1,0,1], 'severity':[0,1,2,1,2]}
df = pd.DataFrame(data)

model = RandomForestClassifier()
model.fit(df[['speed','weather']], df['severity'])

# Test
result = model.predict([[70,1]])
print(f"Predicted Severity for speed 70: {result[0]}")
