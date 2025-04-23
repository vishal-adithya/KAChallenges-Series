#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 05:57:43 2025

@author: vishaladithyaa
"""

import os
import pandas as pd
import spacy

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")



df = pd.read_csv(TRAIN_DF_FILEPATH)
print(df.isnull().sum())

nlp = spacy.load("en_core_web_sm")

math_entities = {
    "ALGEBRA": [
        "equation", "variable", "factor", "solve",
        "polynomial", "linear", "quadratic", "simplify"
    ],
    "GEOMETRY_AND_TRIGONOMETRY": [
        "angle", "triangle", "cosine", "sine", "tangent",
        "geometry", "hypotenuse", "radius", "pi", "parallel"
    ],
    "CALCULUS_AND_ANALYSIS": [
        "derivative", "integral", "limit", "function", "slope",
        "continuity", "differentiable", "curve", "differential", "optimization"
    ],
    "PROBABILITY_AND_STATISTICS": [
        "probability", "mean", "median", "variance", "standard deviation",
        "correlation", "hypothesis", "distribution", "random", "regression", "chi-square"
    ],
    "NUMBER_THEORY": [
        "prime", "divisor", "integer", "factor", "greatest common divisor",
        "modular", "congruence", "Fibonacci", "even", "odd"
    ],
    "COMBINATORICS_AND_DISCRETE_MATH": [
        "permutation", "combination", "set", "graph", "tree",
        "subset", "counting", "binomial", "sequence", "factorial"
    ],
    "LINEAR_ALGEBRA": [
        "matrix", "vector", "determinant", "eigenvalue", "eigenvector",
        "linear transformation", "system of equations", "rank", "scalar"
    ],
    "ABSTRACT_ALGEBRA_AND_TOPOLOGY": [
        "group", "ring", "field", "topology", "homomorphism",
        "isomorphism", "symmetry", "manifold", "space", "continuity", "category"
    ]
}



def Preprocessing(text, entities=math_entities):
    processed = []
    doc = nlp(text.lower())
    
    for token in doc:
        lemma_text = token.lemma_
        tagged = False
        
        for category, terms in entities.items():
            if lemma_text in terms:
                processed.append(f"[{category}:{lemma_text}]")
                tagged = True
                break
        
        if not tagged:
            processed.append(lemma_text)
    
    return " ".join(processed)

df["preprocessed_question"] = df["Question"].apply(Preprocessing)