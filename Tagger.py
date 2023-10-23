import csv
import re
import unidecode as ud
import pandas as pd
import numpy as np
import spacy
import spacy_cleaner
import string
import xml.etree.ElementTree as ET
from spacy_cleaner.processing import removers
from fuzzywuzzy import fuzz


nlp = spacy.load("en_core_web_sm")


def main():
    #path = 'do_sparql_results(1).csv'
    #new_df = edit_disease_df(path)
    #edit_gene_df('Data/Gene/Pmidlist.Test.txt')

    diseases = create_disease_dict(pd.read_csv("Data/Disease/edited_data.csv", header=0))
    #genes=create_gene_dict(pd.read_csv"Data/Gene/gene2go",header=0,columns=['#tax_id', 'GeneID','GO_ID','Evidence','Qualifier', 'GO_term PubMed', 'Category'])

    df = pd.read_csv('Data/Disease/NCBIdevelopset_corpus.csv', sep=';', header=0)
    #df2=pd.read_csv('Data/Gene/outgenes.csv',header=True)

    #for both the disease and gene tagger perform preprocessing over text and only use exact matching for the genetagger
    abstracts = df[df["TYPE"] == 'a']
    exact=[]
    fuzzy=[]
    total = 0
    for abstract in abstracts["TEXT"]:
        words = preprocess(abstract)
        #abstract1.append(abstract)
        total = len(words) + total
        exact.append(exactmatching(words, diseases))
        fuzzy.append(fuzzymatching(words, diseases))


    #new_df = pd.DataFrame(list(zip(*[abstract1, exact, fuzzy]))).add_prefix('Col')
    #new_df.to_csv('outdiseases.csv', index=False)
    #print(total)


def edit_gene_df(path):
    data=pd.read_csv(path)
    df = pd.DataFrame(data,columns=['tags'])
    df = df.reset_index()
    new_df =pd.DataFrame(columns=["ID","TEXT"])
    for i in range(len(df)):
        info=str(df.loc[i,'tags'])
        location = 'Data/Gene/FINAL/'+info+'.BioC.XML'
        tree=ET.parse(location)
        root=tree.getroot()
        for inf in root.iter('text'):
            new_df.loc[len(new_df.index)]=[i,inf.text]
    new_df.to_csv('outgenes.csv', index=True)


def edit_disease_df(path):
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

def create_gene_dict(df):
    # creating the dictionary based on parents and synonyms, with the key as synonym, so it points to the parent
    df.groupby('Evidence').agg({'GeneID': lambda x: ' '.join(x.dropna())})
    df.drop_duplicates(subset=['Evidence', 'GeneID'], inplace=True)
    genes = df.groupby('Evidence')['GeneID'].agg(list).to_dict()

    return genes

def create_disease_dict(df):
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
    #print(matches)
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
            #print(word)
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
