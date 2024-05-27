import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV file
file_path = 'C:/Users/kimdo/Desktop/wCIS/stats_2002to2023.csv'
data = pd.read_csv(file_path)

# Select relevant columns for features and target
features = ['AVG', 'OBP', 'SLG', 'OPS', 'RBI', 'R', 'H', 'D', 'T', 'HR', 'TB', 'SAC', 'SF', 'BB', 'SO', 'GDP', 'MH', 'RISP', 'PH_BA']

# Ensure the RBI column exists
if 'RBI' not in data.columns:
    data['RBI'] = 0

# Separate the features and target
X = data[features].copy()
y = data[features].copy()

# Standardize the feature columns for training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the target score using the initial weights (for initial ranking, if needed)
initial_weights = {
    'AVG': 4,
    'OBP': 2,
    'SLG': 2,
    'OPS': 5,
    'RBI': 3,
    'R': 2,
    'H': 4,
    'D': 2,
    'T': 3,
    'HR': 3,
    'TB': 2,
    'SAC': 2,
    'SF': 2,
    'BB': 3,
    'SO': -2,
    'GDP': -4,
    'MH': 2,
    'RISP': 3,
    'PH_BA': 3,
}
data['Initial_Score'] = (
    initial_weights['AVG'] * data['AVG'] +
    initial_weights['OBP'] * data['OBP'] +
    initial_weights['SLG'] * data['SLG'] +
    initial_weights['OPS'] * data['OPS'] +
    initial_weights['RBI'] * data['RBI'] +
    initial_weights['R'] * data['R'] +
    initial_weights['H'] * data['H'] + 
    initial_weights['D'] * data['D'] +
    initial_weights['T'] * data['T'] +
    initial_weights['HR'] * data['HR'] +
    initial_weights['TB'] * data['TB'] +
    initial_weights['SAC'] * data['SAC'] +
    initial_weights['SF'] * data['SF'] +
    initial_weights['BB'] * data['BB'] +
    initial_weights['SO'] * data['SO'] +
    initial_weights['GDP'] * data['GDP'] +
    initial_weights['MH'] * data['MH'] +
    initial_weights['RISP'] * data['RISP'] +
    initial_weights['PH_BA'] * data['PH_BA']
) / 10

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['Initial_Score'], test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss}')

# Predict the scores using the trained model
X_scaled_full = scaler.transform(X)
learned_scores = model.predict(X_scaled_full).flatten()
data['Learned_Score'] = pd.Series(learned_scores, index=data.index)

# Rename 'Learned_Score' to 'wCIS'
data.rename(columns={'Learned_Score': 'wCIS'}, inplace=True)

# Rank players based on the learned score
ranked_data = data.copy()
ranked_data = ranked_data.sort_values(by='wCIS', ascending=False)

# Reorder the columns to show the learned score
ranked_data.drop(columns=['Initial_Score'], inplace=True)
ranked_data.insert(3, 'wCIS', ranked_data.pop('wCIS'))

# Save the ranked data to a new CSV file, including original features and learned scores
output_file_path = 'C:/Users/kimdo/Desktop/wCIS/wCIS.csv'  # 적절한 파일 경로로 변경
ranked_data.to_csv(output_file_path, index=False)

# Display the top 5 players
print(ranked_data.head())
