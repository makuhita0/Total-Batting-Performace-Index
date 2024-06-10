import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

try:
    standard_data = pd.read_csv('stats/standard.csv')
except FileNotFoundError:
    print("Error: 'standard.csv' 파일을 찾을 수 없음.") 
    exit()  

try:
    stats_data = pd.read_csv('stats/stats_2002to2023.csv')
except FileNotFoundError:
    print("Error: 'stats_2002to2023.csv' 파일을 찾을 수 없음.")
    exit()

features = ['AVG', 'OBP', 'SLG', 'OPS', 'RBI', 'R', 'H', 'D', 'T', 'HR', 'TB', 'SAC', 'SF', 'BB', 'SO', 'GDP', 'MH', 'RISP', 'PH_BA']

X_train = standard_data[features].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test = stats_data[features].copy()
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, X_train_scaled, epochs=1000, validation_split=0.2, verbose=1)

learned_scores = model.predict(X_test_scaled).flatten()
stats_data['Learned_Score'] = pd.Series(learned_scores, index=stats_data.index)

stats_data.rename(columns={'Learned_Score': 'TBPI'}, inplace=True)

ranked_data = stats_data.copy()
ranked_data = ranked_data.sort_values(by='TBPI', ascending=False)

output_file_path = 'TBPI.csv' 
ranked_data.to_csv(output_file_path, index=False)

print(ranked_data.head())
