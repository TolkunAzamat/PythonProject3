import re

# 🔤 Суффикстердин сөздүк
derivational_suffixes = {
    "дөй": "Derivation=Simil", "дай": "Derivation=Simil", "дей": "Derivation=Simil", "дой": "Derivation=Simil",
    "луу": "Derivation=AdjFromNoun", "туу": "Derivation=AdjFromNoun", "дүү": "Derivation=AdjFromNoun", "сыз": "Derivation=AdjFromNoun",
    "чыл": "Derivation=NounFromAdj", "чы": "Derivation=Agent", "чи": "Derivation=Agent",
    "лык": "Derivation=Abstr", "дук": "Derivation=Abstr", "дик": "Derivation=Abstr", "түк": "Derivation=Abstr",
    "таш": "Derivation=VerbFromNoun",
    "ган": "VerbForm=Part", "ген": "VerbForm=Part", "кан": "VerbForm=Part", "кен": "VerbForm=Part",
    "уучу": "Derivation=Agent", "үүчү": "Derivation=Agent",
    "ма": "Derivation=NounFromVerb", "ме": "Derivation=NounFromVerb", "ба": "Derivation=NounFromVerb", "бе": "Derivation=NounFromVerb"
}


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def guess_derivation(word, lemma):
    matched = [tag for suf, tag in derivational_suffixes.items() if word.endswith(suf)]
    return matched[0] if matched else "_"


def tokenize(sentence):
    # дефис менен татаал сөздөрдү бөлбөйт
    return re.findall(r'\w+(?:-\w+)*|[^\w\s]', sentence, re.UNICODE)


def annotate_sentence(sentence, sent_id, file_handle):
    words = tokenize(sentence)
    file_handle.write(f"# sent_id = {sent_id}\n")
    file_handle.write(f"# text = {sentence}\n")

    i = 0
    token_id = 1
    while i < len(words):
        # ЭКИ сөздүк айкашы
        if i < len(words) - 1:
            w1, w2 = words[i], words[i+1]
            print(f"👉 Бул 2 сөз бир мааниде колдонулганбы: “{w1} {w2}”? [Y/n]: ", end="")
            answer = input().lower()
            if answer in ["y", ""]:
                lemma1 = input(f"  Лемма '{w1}' (Enter = {w1.lower()}): ") or w1.lower()
                upos1 = input(f"  Теги '{w1}': ").upper()
                feats1 = guess_derivation(w1.lower(), lemma1)

                lemma2 = input(f"  Лемма '{w2}' (Enter = {w2.lower()}): ") or w2.lower()
                upos2 = input(f"  Теги '{w2}': ").upper()
                feats2 = guess_derivation(w2.lower(), lemma2)

                file_handle.write(f"{token_id}-{token_id+1}\t{w1} {w2}\t_\tMWE\t_\t_\t_\t_\t_\t_\n")
                file_handle.write(f"{token_id}\t{w1}\t{lemma1}\t{upos1}\t_\t{feats1}\t_\t_\t_\t_\n")
                file_handle.write(f"{token_id+1}\t{w2}\t{lemma2}\t{upos2}\t_\t{feats2}\t_\t_\t_\t_\n")
                i += 2
                token_id += 2
                continue

        # Жалгыз сөз
        word = words[i]
        lemma = input(f"Сөз: {word} | Лемма (Enter = {word.lower()}): ") or word.lower()
        upos = input(f"Теги (мис: NOUN, VERB, ADJ, ...): ").upper()
        feats = guess_derivation(word.lower(), lemma)
        file_handle.write(f"{token_id}\t{word}\t{lemma}\t{upos}\t_\t{feats}\t_\t_\t_\t_\n")
        token_id += 1
        i += 1

    file_handle.write("\n")


def main():
    input_file = "C:\\Users\\user\\PycharmProjects\\PythonProject3\\kyrgyz_text.txt"
    output_file = "kyrgyz_corpus.conllu"

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, sentence in enumerate(sentences, 1):
            print(f"\n=== Сүйлөм {i} ===")
            annotate_sentence(sentence, sent_id=i, file_handle=f_out)

    print(f"\n✅ Corpus сакталды: {output_file}")
    print(f"📊 Жалпы сүйлөмдөр: {len(sentences)}")


if __name__ == "__main__":
    main()
