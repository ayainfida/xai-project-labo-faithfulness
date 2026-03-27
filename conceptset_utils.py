import re
import math
import clip
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer    

"""
Adapted from Label-free CBMs https://github.com/Trustworthy-ML-Lab/Label-free-CBM
"""
    
def remove_too_long(concepts, max_len, print_prob=0):
    """
    deletes all concepts longer than max_len
    """
    new_concepts = []
    for concept in concepts:
        if len(concept) <= max_len:
            new_concepts.append(concept)
        else:
            if random.random()<print_prob:
                print(len(concept), concept)
    print(len(concepts), len(new_concepts))
    return new_concepts


def filter_too_similar_to_cls(concepts, classes, sim_cutoff, device="cuda", print_prob=0):
    #first check simple text matches
    print(len(concepts))
    concepts = list(concepts)
    concepts = sorted(concepts)
    
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix+cls)
                if random.random()<print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix+cls))
            except(ValueError):
                pass
        try:
            concepts.remove(cls.upper())
        except(ValueError):
            pass
        try:
            concepts.remove(cls[0].upper()+cls[1:])
        except(ValueError):
            pass
    print(len(concepts))
        
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clip_dot_prods(classes, concepts)
    #weighted since mpnet has highger variance
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if j not in to_delete:
                    to_delete.append(j)
                    if random.random()<print_prob:
                        print("Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(classes[i], concepts[j], dot_prods[i,j], concepts[j]))
                        print("".format(concepts[j]))
                        
    to_delete = sorted(to_delete)[::-1]

    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts

def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):
    
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clip_dot_prods(concepts, concepts)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    #Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[i]))
                    else:
                        to_delete.append(j)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[j]))
                            
    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def _clip_dot_prods(list1, list2, device="cuda", clip_name="ViT-B/16", batch_size=500):
    "Returns: numpy array with dot products"
    clip_model, _ = clip.load(clip_name, device=device)
    text1 = clip.tokenize(list1).to(device)
    text2 = clip.tokenize(list2).to(device)
    
    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1)/batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2)/batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)
        
    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()

"""
LaBo Style concept filtering pipeline:
"""

def normalize_text(text):
    """
    Lowercase and collapse whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    """
    Tokenize into word tokens.
    """
    return re.findall(r"\b\w+\b", normalize_text(text))


def clean_concept_list(concepts, class_names, super_class = "object", verbose = True):
    """
    1. If an exact class name appears in the concept, we replace it with super_class.
    2. For multi-token class names, if all class-name tokens appear in the concept, we delete the concept
    3. Remove the word 'overall' from concepts, as gemini output contains it in the 10th sentence.
    """
    cleaned = []
    replaced_count = 0
    deleted_count = 0

    super_class_norm = normalize_text(super_class)
    class_names_norm = [normalize_text(c) for c in class_names]

    for concept in concepts:
        original_concept = concept
        concept = normalize_text(concept)
        concept = re.sub(r"\boverall, \b", "", concept)
        concept = re.sub(r"\s+", " ", concept).strip()
        if not concept:
            continue

        replaced_classes = set()

        # Heuristic 1: exact phrase replacement
        for class_name_norm in class_names_norm:
            pattern = r"\b" + re.escape(class_name_norm) + r"\b"

            if re.search(pattern, concept):
                new_concept = re.sub(pattern, super_class_norm, concept)
                if new_concept != concept:
                    concept = re.sub(r"\s+", " ", new_concept).strip()
                    replaced_classes.add(class_name_norm)
                    replaced_count += 1
                    if verbose:
                        print(f"  [REPLACED] '{original_concept}' -> '{concept}'")

        # Heuristic 2: delete if all tokens of a multi-token class name are present
        concept_tokens = set(tokenize(concept))
        keep_concept = True

        for class_name_norm in class_names_norm:
            if class_name_norm in replaced_classes:
                continue

            class_tokens = tokenize(class_name_norm)
            if len(class_tokens) <= 1:
                continue

            if all(tok in concept_tokens for tok in class_tokens):
                keep_concept = False
                deleted_count += 1
                if verbose:
                    print(f"  [DELETED]  '{original_concept}' (contains all tokens of '{class_name_norm}')")
                break

        # Additional rule: if concept collapses to only the super class, drop it
        if keep_concept and concept == super_class_norm:
            keep_concept = False
            deleted_count += 1
            if verbose:
                print(f"  [DELETED]  '{original_concept}' (collapsed to super class '{super_class_norm}')")

        if keep_concept and concept:
            cleaned.append(concept)

    if verbose:
        print("\n--- Summary ---")
        print(f"Original concepts: {len(concepts)}")
        print(f"Replaced: {replaced_count}")
        print(f"Deleted: {deleted_count}")
        print(f"Cleaned concepts: {len(cleaned)}")

    return cleaned


def clean_dataset_one_class_at_a_time(data_json, dataset_key,super_class = "object", verbose = False):
    """
    Clean concepts class-by-class while keeping the same structure:
    data_json[dataset_key][class_name]["extracted_concepts"]
    """
    class_names = list(data_json[dataset_key].keys())

    for class_name, payload in data_json[dataset_key].items():
        raw_concepts = payload.get("concepts", [])
        if verbose:
            print(f"\n[{dataset_key}/{class_name}] cleaning {len(raw_concepts)} concepts")

        payload["extracted_concepts"] = clean_concept_list(
            concepts=raw_concepts,
            class_names=class_names,
            super_class=super_class,
            verbose=verbose,
        )

    return data_json