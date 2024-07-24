# 데이터 전처리
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/total_analysis.csv")

# datetime형식을 코드 실행 시 정수로 변환하면 에러 발생
# 전처리 단계에서 미리 변환
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data = data.drop(columns=['date'])

# Extract features and target
features = data.drop(columns=['error'])
target = data['error']

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into train and test sets with 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, shuffle=False)

# Define a function to create sequences
def create_sequences(data, target, sequence_length):
    sequences = []
    target_values = []
    
    data = np.array(data)
    target = np.array(target)

    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        target_values.append(target[i + sequence_length])
    
    return np.array(sequences), np.array(target_values)

# Create sequences
sequence_length = 10
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# Define model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_seq, y_train_seq, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2  # Use 20% of the training data for validation
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')


# ===========================================================
# 학습과정의 시각화
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()

plot_training_history(history)

#모델 평가 결과 요약
from sklearn.metrics import precision_score, recall_score, f1_score

# Predictions
y_pred = (model.predict(X_test_seq) > 0.5).astype("int32")

# Evaluation metrics
precision = precision_score(y_test_seq, y_pred)
recall = recall_score(y_test_seq, y_pred)
f1 = f1_score(y_test_seq, y_pred)

# Print summary
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
