import spacy
import pickle

nlp = spacy.load("en_core_web_sm")
sentence = "Why do you think it happened"

SRC = pickle.load(open(f'weights/SRC.pkl', 'rb'))

for tok in nlp.tokenizer(sentence):
    if tok.text != " ":
        print(tok.text, SRC.vocab.stoi[tok.text])