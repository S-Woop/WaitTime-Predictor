import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv('waitlist_data.csv')

# Preprocess the data
data['Holiday'] = data['Holiday'].astype(int)
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'])
data['DateTime'] = data['Date'] + data['Time'].dt.time

# Split the data into training and testing sets
X = data[['Holiday', 'DateTime', 'Number']]
y = data['SeatingTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Predict the seating time for new inputs
new_inputs = pd.DataFrame({
    'Holiday': [0],
    'DateTime': ['2022-12-25 18:00:00'],
    'Number': [5]
})
predicted_seating_time = model.predict(new_inputs)
print(f'Predicted Seating Time: {predicted_seating_time}')