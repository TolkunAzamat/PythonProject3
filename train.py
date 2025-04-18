import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


def load_conllu_data(filepath):
    words = []
    upos_tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split('\t')
            if len(parts) == 10:
                word = parts[1].lower()
                upos = parts[3]
                words.append(word)
                upos_tags.append(upos)
    return words, upos_tags


def train_model(X, y):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
    X_vect = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vect, y)

    return clf, vectorizer


def main():
    conllu_path = "kyrgyz_corpus.conllu"
    model_path = "upos_model.pkl"
    vectorizer_path = "upos_vectorizer.pkl"

    print("📚 Чтение корпуса...")
    X, y = load_conllu_data(conllu_path)

    print(f"🔤 Слов: {len(X)}, Тегов: {len(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Обучение модели...")
    model, vectorizer = train_model(X_train, y_train)

    print("🧪 Оценка...")
    X_test_vect = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vect)
    print(classification_report(y_test, y_pred))

    print(f"💾 Сохранение модели в: {model_path}")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("✅ Готово!")


if __name__ == "__main__":
    main()
