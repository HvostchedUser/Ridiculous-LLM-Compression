import json
import numpy as np

def diff_to_importances(diffs, criteria='abs'):
    scores = [[k, np.mean(i)] for k, i in diffs[criteria].items()]
    return sorted(scores, key=lambda x: x[1])

def calc_mean_abs_diff(a0, a1):
    delta = np.abs(a0 - a1)
    delta[delta==np.inf] = np.nan
    return np.nanmean(delta)

def calc_mean_std_dim(a0, a1):
    delta = a0 - a1s
    delta[np.abs(delta)==np.inf] = np.nan
    return np.mean(np.nanstd(r, axis=1))

def cosine_vectors(vec1, vec2):
    mask = ~np.isinf(vec1) & ~np.isinf(vec2)
    vec1_filtered = vec1[mask]
    vec2_filtered = vec2[mask]
    dot_product = np.dot(vec1_filtered, vec2_filtered)
    norm_vec1 = np.linalg.norm(vec1_filtered)
    norm_vec2 = np.linalg.norm(vec2_filtered)
    if norm_vec1 == 0 or norm_vec2 == 0 or not(len(vec1_filtered)):
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)    

def calc_mean_cosine(a0, a1, shape):
    a0[np.isnan(a0)] = np.inf
    a1[np.isnan(a1)] = np.inf
    if max(np.max(np.abs(a0)), np.max(np.abs(a1))) == np.inf:
        vals = []
        for i in range(len(a0)):
            vals.append(cosine_vectors(a0[i], a1[i]))
        return np.mean(vals)
    A = a0.reshape(shape)
    B = a1.reshape(shape)
    return np.mean(np.sum(A * B, axis=1) 
                   / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)))

# def calc_imps(hiddens, 

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_path", type=str, required=True,
                       help="Path to json with hiddens")
    parser.add_argument("--layer_type", type=str, required=True,
                       help="One of layers ('mlp' is the best) to measure between this decoder layer elements: 'input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'")
    args = parser.parse_args()
    
    with open(args.hidden_path, 'r') as f:
      hiddens = json.load(f)

    n_layers = len(hiddens[args.layer_type])
    n_questions = len(hiddens[args.layer_type]['0'])
    
    diffs = {i: {j: [] for j in range(n_layers-1)} for i in ['abs', 'std', 'cos']}
    
    for n in range(n_layers-1):
        for q in range(n_questions):
            prev = np.array(hiddens[args.layer_type][str(n)]['hidden_states'][q])
            cur = np.array(hiddens[args.layer_type][str(n+1)]['hidden_states'][q])
            shape = hiddens[args.layer_type][str(n)]['shapes'][q]
            diffs['abs'][n].append(calc_mean_abs_diff(cur, prev))
            diffs['std'][n].append(calc_mean_abs_diff(cur, prev))
            diffs['cos'][n].append(1 - calc_mean_cosine(cur, prev, shape))

    return diff_to_importances(diffs, 'cos')



if __name__ == "__main__":
    print(main())

    
