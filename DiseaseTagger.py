import re
import unidecode as ud
import pandas as pd
import numpy as np
import spacy
import spacy_cleaner
import string
from spacy_cleaner.processing import removers
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

nlp = spacy.load("en_core_web_sm")


def main():
    path = 'do_sparql_results(1).csv'
    new_df = edit_df(path)
    diseases = create_dict(new_df)
    # example
    # key = 'SPG14'
    # if key in diseases.keys():
    #   print(f"yes, we have found the disease: '{key}'and the disease it points to is '{ diseases.get(key) }'")
    # else:
    #   print(f"'{key}' does not exist in this database")
    abstract = "Alzheimer’s disease (AD) is a complex and heterogeneous neurodegenerative disorder, classified as " \
               "either early onset (under 65 years of age), or late onset (over 65 years of age). Three main genes " \
               "are involved in early onset AD: amyloid precursor protein (APP), presenilin 1 (PSEN1), and presenilin " \
               "2 (PSEN2). The apolipoprotein E (APOE) E4 allele has been found to be a main risk factor for " \
               "late-onset Alzheimer’s disease. Additionally, genome-wide association studies (GWASs) have identified " \
               "several genes that might be potential risk factors for AD, including clusterin (CLU), complement " \
               "receptor 1 (CR1), phosphatidylinositol binding clathrin assembly protein (PICALM), " \
               "and sortilin-related receptor (SORL1). Recent studies have discovered additional novel genes that " \
               "might be involved in late-onset AD, such as triggering receptor expressed on myeloid cells 2 (TREM2) " \
               "and cluster of differentiation 33 (CD33). Identification of new AD-related genes is important for " \
               "better understanding of the pathomechanisms leading to neurodegeneration."
    words = preprocess(abstract)
    print(exactmatching(words, diseases))
    # print(fuzzymatching(words, diseases))


def edit_df(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data, columns=['id', 'label', 'parent', 'synonym'])
    df['synonym'] = df['synonym'].replace('', np.nan)
    df['length'] = df.synonym.str.len()
    df = df[df.length > 3]
    df['synonym'] = df['synonym'].apply(normalize_string)
    #df['label'] = df['label'].apply(normalize_string)
    df.to_csv('edited_data.csv')
    return df


def normalize_string(word):
    ua_disease = ud.unidecode(word)
    lc_disease = ua_disease.lower()
    pc_disease = remove_punctuation(lc_disease)
    rmv_disease = pc_disease.replace(" disease", "")
    rmv2_disease = rmv_disease.replace(" syndrome", "")
    return rmv2_disease

def remove_punctuation(input_string):
    # Make a regular expression that matches all punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # Use the regex
    return regex.sub('', input_string)

def create_dict(df):
    # creating the dictionary based on parents and synonyms, with the key as synonym, so it points to the parent
    df.groupby('synonym').agg({'label': lambda x: ' '.join(x.dropna())})
    df.drop_duplicates(subset=['synonym', 'label'], inplace=True)
    diseases = df.groupby('synonym')['label'].agg(list).to_dict()

    df.groupby('label').agg({'label': lambda x: ' '.join(x.dropna())})
    df.drop_duplicates(subset=['label', 'label'], inplace=True)
    diseases_2 = df.groupby('label')['label'].agg(list).to_dict()
    diseases_3 = {**diseases_2, **diseases}
    return diseases_3


def preprocess(abstract):
    pipeline = spacy_cleaner.Pipeline(nlp, removers.remove_punctuation_token, removers.remove_stopword_token)
    pipeline.clean(abstract)
    full_text = nlp(abstract)
    tokens = []
    # example text for regex pattern recognition
    # abstract = abstract + "Alzheimer's disease and Parkinson's disease are neurological disorders."

    # pattern = r"\b\w+’s disease\b|\b\w+'s disease\b|\b\w+syndrome|\b\w+cancer"
    # matches = re.findall(pattern, abstract)
    # for match in matches:
    #     match = match.replace("’", "'")
    #     tokens.append(match)

    for sent in full_text.sents:
        for token in sent:
            tokens.append(token.lemma_)
    return tokens


def exactmatching(words, diseases):
    diseaseslist = []
    for word in words:
        if word in diseases.keys():
            diseaseslist.append(diseases.get(word))
            print(diseases.get(word))
    return diseaseslist


def fuzzymatching(words, diseases):
    # Using FuzzyWuzzy for the fuzzy matching, which uses Levenshtein's distance
    # it seems to also register the token 's and diseases as a match
    MIN_MATCH_SCORE = 99
    guessed_word = []
    for word in words:
        for key, value in diseases.items():  # dict.items used to access the (key-value) pair of dictionary
            if (fuzz.token_set_ratio(word, key) >= MIN_MATCH_SCORE):  # check the condition for each value,
                # if it's satisfied create a dictionary & append it to result list
                print(word, key)
                dict = {}
                dict[key] = value
                guessed_word.append(dict)

    print(guessed_word)


main()
