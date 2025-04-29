import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib


# 1. Загрузка размеченного корпуса
def load_data(excel_path):
    df = pd.read_excel(excel_path)  # Используйте read_excel для файлов .xlsx
    data = df[["FORM", "LEMMA", "UPOS"]]  # Подразумевается, что столбцы имеют такие имена

    # Удаляем строки с пропущенными значениями в столбце "UPOS"
    data = data.dropna(subset=["UPOS"])
    return data


# 2. Разделение данных на обучающую и тестовую выборки
def split_data(data):
    X = data["FORM"].values  # Токены
    y = data["UPOS"].values  # POS теги
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Токенизация и векторизация
def vectorize_data_with_ngrams(X_train, X_test):
    X_train = pd.Series(X_train).astype(str).fillna('')
    X_test = pd.Series(X_test).astype(str).fillna('')

    vectorizer = CountVectorizer(ngram_range=(1, 2))  # Используем биграммы
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec


# 4. Обучение модели
def train_model(X_train_vec, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_train)
    return clf


# 5. Оценка модели
def evaluate_model(clf, X_test_vec, y_test):
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))


# 6. Сохранение модели
def save_model(clf, vectorizer, model_path, vectorizer_path):
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)


# 7. Использование модели для предсказаний
def predict_new_data(clf, vectorizer, new_text):
    new_text_vec = vectorizer.transform(new_text)
    predictions = clf.predict(new_text_vec)
    return predictions


# 2.1. Балансировка классов с помощью SMOTE
def balance_classes(X_train_vec, y_train):
    # Проверка минимального количества образцов в каждом классе
    class_counts = Counter(y_train)
    min_class_count = min(class_counts.values())

    if min_class_count <= 1:
        raise ValueError("SMOTE не может работать с классами, содержащими только 1 пример.")

    # Устанавливаем k_neighbors в зависимости от самого маленького класса
    k_neighbors = min(5, min_class_count - 1)

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)

    print(f"Before SMOTE, class distribution: {Counter(y_train)}")
    print(f"After SMOTE, class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled


# Фильтрация классов с малым количеством примеров
def filter_classes(data, min_samples=2):
    # Фильтруем классы, у которых меньше чем min_samples примеров
    class_counts = Counter(data["UPOS"])
    filtered_data = data[data["UPOS"].isin([k for k, v in class_counts.items() if v >= min_samples])]
    return filtered_data


# Основной процесс
def main():
    # Загрузка данных
    data = load_data("C:\\Users\\user\\Downloads\\pos_annotation_template_ky.xlsx")

    # Фильтрация классов с очень малым числом примеров
    data = filter_classes(data, min_samples=2)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(data)

    # Векторизация текста
    vectorizer, X_train_vec, X_test_vec = vectorize_data_with_ngrams(X_train, X_test)

    # Балансировка классов
    X_train_vec, y_train = balance_classes(X_train_vec, y_train)

    # Обучение модели
    clf = train_model(X_train_vec, y_train)

    # Оценка модели
    evaluate_model(clf, X_test_vec, y_test)

    # Сохранение модели и векторизатора
    save_model(clf, vectorizer, 'pos_tagging_model.pkl', 'vectorizer.pkl')

    # Пример использования модели для предсказания
    new_text = ["Согуш начался."]
    predictions = predict_new_data(clf, vectorizer, new_text)
    print("Predictions for new text:", predictions)


if __name__ == "__main__":
    main()
