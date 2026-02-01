import numpy as np
from collections import Counter
import math
def generate_i_gram(sentence: list[str], i:int):
    window = []
    st, end, n = 0,0, len(sentence)
    i_gram = []
    while end < n :
        # print(window)
        window.append(sentence[end])
        while (end - st + 1) > i:
            window = window[1:]
            st += 1
        if (end-st + 1) == i:
          i_gram.append(tuple(window))
        end += 1
        
    return Counter(i_gram)
        
def find_max_r_refs(key, references_dict):
    max_gram = 0 
    for ref in references_dict:
        if key in ref:
            max_gram = max(max_gram, ref[key])
    return max_gram
def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max