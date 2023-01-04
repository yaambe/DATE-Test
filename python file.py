import re
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from tqdm import tqdm
import os
import pickle


with open('hs_model.pkl', 'rb') as f:
    model = pickle.load(f)

hs_info_df = pd.read_pickle("hs_info_df.pickle")

space_pattern = re.compile(r'\s\s+') # Process whitespace
remove_pattern = re.compile(r'[\;\,\)\(\[\]]') # select only ; , ) ( ] [

org_stop_words = {'of', 'or', 'and', 'for', 'than', 'the', 'in', 'with', 'to', 'but', 'by', 'whether',
                  'on', 'its', 'an', 'their', 'at', 'this', 'which', 'from', 'as', 'be', 'is', 'other'}

# Create a list where the preprocessing data is stored.
refined_text_list = []

def simil_test_input(model, input_text):
    # Preprocessing
    input_text = re.sub(space_pattern, " ", input_text)
    input_text = input_text.lower()
    input_text = re.sub(remove_pattern, " ", input_text)
    text_split = input_text.split()
    org_contents_split = [w for w in text_split if len(w.strip()) > 1 and w.strip() not in org_stop_words]  
    org_contents = " ".join(w for w in org_contents_split)
    final_sentence = re.sub(space_pattern, " ", org_contents)
    refined_text_list = final_sentence.split()
    # Model input
    input_vector = model.infer_vector(refined_text_list, alpha=0.025, min_alpha=0.001)
    similarities = model.docvecs.most_similar([input_vector], topn=3)
    simil_result = [(simil[0].replace('TARGET_', ''), round(simil[1], 4)) for simil in similarities]
    return simil_result

for i in range(1,1000):
    input_text = input("DESC input:")
    result = simil_test_input(model, input_text)

    for idx, data in enumerate(result):
        this_name_df = hs_info_df[hs_info_df['HSCODE'] == data[0]]
        print("\n{}rank :\n\tHSCODE : {}\n\tinformation : {}\n\tSCORE : {}".format(idx + 1, data[0], this_name_df['information'].values[0], data[1]))
