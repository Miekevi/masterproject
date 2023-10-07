import re
import unidecode as ud
import pandas as pd
import numpy as np
import spacy
import spacy_cleaner
import string
from spacy_cleaner.processing import removers
from fuzzywuzzy import fuzz, process

nlp = spacy.load("en_core_web_sm")


def main():
    #path = 'do_sparql_results(1).csv'
    #new_df = edit_df(path)
    diseases = create_dict(pd.read_csv("edited_data.csv", header=0))
    df = pd.read_csv('NCBIdevelopset_corpus.csv', sep=';', header=0)
    abstracts = df[df["TYPE"] == 'a']
    for abstract in abstracts["TEXT"]:
        words = preprocess(abstract)
        print(exactmatching(words, diseases))
        print(fuzzymatching(words, diseases))
    #abstract = "The common hereditary forms of breast cancer have been largely attributed to the inheritance of mutations in the BRCA1 or BRCA2 genes. However, it is not yet clear what proportion of hereditary breast cancer is explained by BRCA1 and BRCA2 or by some other unidentified susceptibility gene (s). We describe the proportion of hereditary breast cancer explained by BRCA1 or BRCA2 in a sample of North American hereditary breast cancers and assess the evidence for additional susceptibility genes that may confer hereditary breast or ovarian cancer risk. Twenty-three families were identified through two high-risk breast cancer research programs. Genetic analysis was undertaken to establish linkage between the breast or ovarian cancer cases and markers on chromosomes 17q (BRCA1) and 13q (BRCA2). Mutation analysis in the BRCA1 and BRCA2 genes was also undertaken in all families. The pattern of hereditary cancer in 14 (61%) of the 23 families studied was attributed to BRCA1 by a combination of linkage and mutation analyses. No families were attributed to BRCA2. Five families (22%) provided evidence against linkage to both BRCA1 and BRCA2. No BRCA1 or BRCA2 mutations were detected in these five families. The BRCA1 or BRCA2 status of four families (17%) could not be determined. BRCA1 and BRCA2 probably explain the majority of hereditary breast cancer that exists in the North American population. However, one or more additional genes may yet be found that explain some proportion of hereditary breast cancer."
    #words = preprocess(abstract)
    #print(exactmatching(words, diseases))
    #print(fuzzymatching(words, diseases))


def edit_df(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data, columns=['id', 'label', 'parent', 'synonym'])
    df['synonym'] = df['synonym'].replace('', np.nan)
    df['length'] = df.synonym.str.len()
    df = df[df.length > 3]
    df['synonym'] = df['synonym'].apply(normalize_string)
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
    #abstract = abstract + "breastcancer"
    # Regex no longer needed for words like disease and syndrome. I will use it for tumors and cancer as those are specific to bodyparts
    pattern = r"\b\w+\s?cancer\b|\b\w+\s?tumor\b|\b\w+\s?cancers\b|\b\w+\s?tumors\b"
    matches = re.findall(pattern, abstract)
    print(matches)
    for match in matches:
        tokens.append(match)
        abstract.replace(match, "")
    for sent in full_text.sents:
        for token in sent:
            token.lemma_ = token.lemma_.lower()
            tokens.append(token.lemma_)
    return tokens


def exactmatching(words, diseases):
    diseaseslist = []
    for word in words:
        if word in diseases.keys():
            print(word)
            diseaseslist.append(diseases.get(word))
    return diseaseslist


def fuzzymatching(words, diseases):
    # Using FuzzyWuzzy for the fuzzy matching, which uses Levenshtein's distance
    threshold = 95
    diseaseslist = []
    for word in words:
        for key in diseases.keys():  # dict.items used to access the (key-value) pair of dictionary
            similarityscore = fuzz.ratio(word,key)
            if similarityscore >= threshold:  # check the condition for each value,
                diseaseslist.append(diseases.get(key))

    return diseaseslist


main()
