import streamlit as st
import pandas as pd
import joblib
import re

# Цвета для разных тегов
POS_COLORS = {
    'NOUN': '#ffcccb',
    'VERB': '#c3f7c3',
    'ADJ': '#add8e6',
    'ADV': '#f9d9f9',
    'PROPN': '#ffd700',
    'PUNCT': '#eeeeee',
    'NUM': '#ffe4b5',
    'ADP': '#e6e6fa',
    'PRON': '#d3ffce',
    'CONJ': '#c1e1c1',
    'PART': '#f4cccc',
    'X': '#dddddd'
}

# Токенизация
def tokenize_text(text):
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

# Загрузка модели и векторизатора
@st.cache_resource
def load_model():
    clf = joblib.load("pos_tagging_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return clf, vectorizer

# Разметка текста
def tag_text(clf, vectorizer, sentence):
    tokens = tokenize_text(sentence)
    vectors = vectorizer.transform(tokens)
    tags = clf.predict(vectors)
    return pd.DataFrame({'Token': tokens, 'POS': tags})


# Основное приложение
def main():
    st.title("📚 POS-теггер (Кыргыз тили)")

    sentence = st.text_area("Текст киргизиңиз:", "Мугалим сабак берди.")

    if st.button("Анализ кылуу"):
        clf, vectorizer = load_model()
        df = tag_text(clf, vectorizer, sentence)

        # Цветовая таблица
        def highlight_row(row):
            color = POS_COLORS.get(row['POS'], '#ffffff')
            return [f'background-color: {color}'] * len(row)

        st.subheader("📊 Натыйжа таблица түрүндө:")
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

        # Скачать как CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ CSV түрүндө жүктөө", data=csv, file_name="tagged_output.csv", mime='text/csv')


if __name__ == "__main__":
    main()
