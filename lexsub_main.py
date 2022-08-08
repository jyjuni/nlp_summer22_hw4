#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

from collections import defaultdict
# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string 

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    syn_names = set()
    for lex in wn.lemmas(lemma, pos=pos):
        # get all lemmas from the synset for current lex
        lemmas = lex.synset().lemmas()
        # syn_names.update([l.name() for l in lemmas if l.name()!=lemma])
        syn_names.update([' '.join(l.name().split('_')) for l in lemmas if l.name()!=lemma])
    
    return syn_names

def get_extended_candidates(lemma, pos, include_similar_tos=True, include_also_sees=True) -> List[str]:
    # same function as get_candidates(), but also takes into account lemmas in similar_to
    syn_names = set()
    for lex in wn.lemmas(lemma, pos=pos):
        # get all lemmas from the synset for current lex
        s = lex.synset()
        similar_s = [s]
        if include_similar_tos:
            similar_s.extend(s.similar_tos())
        if include_also_sees:
            similar_s.extend(s.also_sees())
            
        for ss in similar_s: 
            lemmas = ss.lemmas()
            # syn_names.update([l.name() for l in lemmas if l.name()!=lemma])
            syn_names.update([' '.join(l.name().split('_')) for l in lemmas if l.name()!=lemma])
    
    return syn_names


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    syn_counts = defaultdict(int)
    
    for lex in wn.lemmas(context.lemma, pos=context.pos):
        # get all lemmas from the synset for current lex
        lemmas = lex.synset().lemmas()
        for l in lemmas:
            if l.name() != context.lemma:
                syn_counts[' '.join(l.name().split('_'))] += l.count()
    
    # sorted_syn = sorted(syn_counts.items(), key = lambda x: x[1], reverse = True)
    # print(sorted_syn)
    
    return sorted(syn_counts.items(), key = lambda x: x[1], reverse = True)[0][0]

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = stopwords.words('english')
    score = {}
    for lex in wn.lemmas(context.lemma, pos=context.pos):
        syn = lex.synset()
        contxt = context.left_context + context.right_context
        gloss = tokenize(syn.definition()) + tokenize(' '.join(syn.examples()))
        
        # add hypernyms' definition and all examples
        for hypernym in syn.hypernyms(): 
            gloss += tokenize(hypernym.definition()) + tokenize(' '.join(hypernym.examples()))
        
        # compute overlap (length of intersection of two word sets) between context and definition+examples
        overlap = [w for w in contxt if w not in stop_words and w in gloss] # remove stopwords
        score[syn] = (len(overlap), lex.count())
    
    syn_sorted = sorted(score.items(), key=lambda x: x[1], reverse=True)
    for syn, _ in syn_sorted:
        candidates = sorted([l for l in syn.lemmas() if l.name()!=context.lemma], key = lambda x: x.count(), reverse = True)
        if candidates:
            return ' '.join(candidates[0].name().split('_'))
    return None
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context : Context) -> str:
        
        def get_similarity(x):
            try:
                sim = self.model.similarity(context.lemma, x)
                # sim = self.model.similarity(context.word_form, x) # use word_form will result in slightly better acc+recall
                return sim
            except KeyError:
                return 0
            
        # possible synonyms from WordNet
        candidates = list(get_candidates(context.lemma, context.pos))
        # print(sorted([(x, get_similarity(x)) for x in candidates], key=lambda x: x[1], reverse = True))
        # find synonym with the highest similarity
        candidates.sort(key=lambda x: get_similarity(x), reverse = True)
            
        return candidates[0] if candidates else 'smurf'


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context, opt=False) -> str:
        if opt:
            candidates = list(get_extended_candidates(context.lemma,context.pos, include_similar_tos=True, include_also_sees=False)) # include similar_tos and exclude also_sees performs best
            # candidates = list(get_extended_candidates(context.lemma,context.pos)) 
        else:
            candidates = list(get_candidates(context.lemma,context.pos))
        
        input_s = ' '.join(context.left_context + ['[MASK]'] + context.right_context)
        input_toks = self.tokenizer.encode(input_s)
        target_i = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][target_i])[::-1] # sort in decreasing order

        for id in best_words:
            tok = self.tokenizer.convert_ids_to_tokens([id])[0]
            if tok in candidates:
                return tok
            
        return 'smurf'

        # result = [tok for tok in self.tokenizer.convert_ids_to_tokens(select_words[:10]) if tok not in ('[SEP]', '[CLS]', '[UNK]')]
        # print(result)
        # print(self.tokenizer.convert_ids_to_tokens(best_words[:10]))
        
if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)
    bert_predictor = BertPredictor()
    
    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context)  #smurf
        # prediction = wn_frequency_predictor(context) #part2
        # prediction = wn_simple_lesk_predictor(context) #part3
        # prediction = predictor.predict_nearest(context) #part4
        # prediction = bert_predictor.predict(context) #part5
        prediction = bert_predictor.predict(context, opt=True) #part6
        
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
