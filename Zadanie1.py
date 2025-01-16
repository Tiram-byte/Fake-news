import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Шаг 1: Загрузка данных
# Предполагается, что у вас есть CSV файл с двумя колонками: 'text' и 'label'
# label должен содержать 'REAL' или 'FAKE'
df = pd.read_csv('fake_news.csv')

# Шаг 2: Подготовка данных
X = df['text']
y = df['label']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Извлечение признаков
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Шаг 4: Обучение модели
classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Шаг 5: Оценка модели
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
