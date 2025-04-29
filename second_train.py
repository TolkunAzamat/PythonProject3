import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib


def load_data(excel_path):
    df = pd.read_excel(excel_path)  # Используйте read_excel для файлов .xlsx
    data = df[["FORM", "LEMMA", "UPOS"]]  # Подразумевается, что столбцы имеют такие имена

    # Удаляем строки с пропущенными значениями в столбце "UPOS"
    data = data.dropna(subset=["UPOS"])
    return data


def preprocess(df):
    X = df['word'].astype(str)
    y = df['pos'].astype(str)
    return X, y


def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def balance_classes(X_train_vec, y_train):
    print("Before SMOTE, class distribution:", Counter(y_train))
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)
    print("After SMOTE, class distribution:", Counter(y_resampled))
    return X_resampled, y_resampled


def train_model(X_train_vec, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)
    return clf


def evaluate(clf, X_test_vec, y_test):
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, zero_division=1))


def predict_new_text(clf, vectorizer, new_words):
    new_vec = vectorizer.transform(new_words)
    predictions = clf.predict(new_vec)
    print("Predictions for new text:", list(predictions))


def main():
    # 1. Загрузка и предобработка данных
    df = load_data("C:\\Users\\user\\Downloads\\pos_annotation_template_ky.xlsx")

    X, y = preprocess(df)

    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Векторизация
    X_train_vec, X_test_vec, vectorizer = vectorize(X_train, X_test)

    # 4. Балансировка классов
    X_train_vec_bal, y_train_bal = balance_classes(X_train_vec, y_train)

    # 5. Обучение модели
    clf = train_model(X_train_vec_bal, y_train_bal)

    # 6. Оценка модели
    evaluate(clf, X_test_vec, y_test)

    # 7. Предсказание нового текста
    new_words = ['китеп', 'мен', 'барам']
    predict_new_text(clf, vectorizer, new_words)

    # 8. Сохраняем модель и векторизатор
    joblib.dump(clf, 'pos_classifier.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


if __name__ == '__main__':
    main()
