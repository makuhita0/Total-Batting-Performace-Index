import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 표준 CSV 파일 로드 (학습용 데이터)
try:
    standard_data = pd.read_csv('standard.csv')
except FileNotFoundError:
    print("Error: 'standard.csv' 파일을 찾을 수 없음.") 
    exit()  

# 통계 CSV 파일 로드 (순위 매기기용 데이터)
try:
    stats_data = pd.read_csv('stats_2002to2023.csv')
except FileNotFoundError:
    print("Error: 'stats_2002to2023.csv' 파일을 찾을 수 없음.")
    exit()

# 특성과 목표로 사용할 열 선택
features = ['AVG', 'OBP', 'SLG', 'OPS', 'RBI', 'R', 'H', 'D', 'T', 'HR', 'TB', 'SAC', 'SF', 'BB', 'SO', 'GDP', 'MH', 'RISP', 'PH_BA']

# 학습용 데이터에서 특성 분리
X_train = standard_data[features].copy()

# 특성 열 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터 (순위 매기기용 데이터)에서 특성 분리
X_test = stats_data[features].copy()
X_test_scaled = scaler.transform(X_test)

# 신경망 모델 구축
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

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train_scaled, X_train_scaled, epochs=1000, validation_split=0.2, verbose=1)

# 학습된 모델을 사용하여 점수 예측
learned_scores = model.predict(X_test_scaled).flatten()
stats_data['Learned_Score'] = pd.Series(learned_scores, index=stats_data.index)

# 'Learned_Score'를 'TBPI'로 변경
stats_data.rename(columns={'Learned_Score': 'TBPI'}, inplace=True)

# 학습된 점수를 기반으로 선수 순위 매기기
ranked_data = stats_data.copy()
ranked_data = ranked_data.sort_values(by='TBPI', ascending=False)

# 원래 특성과 학습된 점수를 포함한 순위 데이터 새로운 CSV 파일로 저장
output_file_path = 'TBPI.csv'  # 적절한 파일 경로로 변경
ranked_data.to_csv(output_file_path, index=False)

# 상위 5명의 선수 표시
print(ranked_data.head())
