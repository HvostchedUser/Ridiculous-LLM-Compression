import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)  # This converts float16, float32, float64 to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Add more if needed
        return super().default(obj)



def write_tokens_to_file(model, questions, tokenizer, generate_tokens=150):
    """Generate responses and return full generation results."""
    results = []
    for text in questions:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generate_tokens,
                do_sample=True,
                temperature=0.2
            )
        decoded = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        results.append({
            "input": text,
            "output": decoded,
            "tokens": outputs[0].cpu().tolist()
        })
    return results


def compute_importances_from_activations(activations, layer_type='mlp', metric='cosine'):
    """
    Compute importance scores between consecutive layers using one of three metrics.

    Args:
        activations: dict of collected hidden states
        layer_type: which sub-layer to analyze (e.g., 'mlp')
        metric: one of {'cosine', 'abs', 'std'}

    Returns:
        List of (layer_name, score) sorted by importance (descending)
    """
    if layer_type not in activations:
        raise ValueError(f"Layer type '{layer_type}' not found. Available: {list(activations.keys())}")

    layers_data = activations[layer_type]
    n_layers = len(layers_data)
    n_questions = len(layers_data[0])

    diffs = {layer: [] for layer in range(n_layers - 1)}

    for n in range(n_layers - 1):
        for q in range(n_questions):
            try:
                prev = np.array(layers_data[n][q])     # shape: [seq_len, d_model]
                cur = np.array(layers_data[n + 1][q])

                # Truncate to same length
                min_len = min(prev.shape[0], cur.shape[0])
                prev = prev[:min_len]
                cur = cur[:min_len]

                if metric == 'cosine':
                    sim = cosine_similarity_vectors(prev, cur)
                    score = 1.0 - sim  # higher = more change

                elif metric == 'abs':
                    score = np.mean(np.abs(cur - prev))

                elif metric == 'std':
                    delta = cur - prev
                    # Compute std along feature dim, then mean over sequence
                    score = np.mean(np.std(delta, axis=1))

                else:
                    raise ValueError(f"Unknown metric: {metric}")

                diffs[n].append(score)

            except Exception as e:
                print(f"Error in layer {n}, question {q}: {e}")
                diffs[n].append(0.0)

    # Average over questions
    scores = [(f"layer_{i}", np.mean(vals)) for i, vals in diffs.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)  # highest first

def cosine_similarity_vectors(vecs1, vecs2):
    """
    Compute mean cosine similarity between two sequences of vectors.
    vecs1, vecs2: (L, D) arrays
    """
    if vecs1.size == 0 or vecs2.size == 0:
        return 0.0

    # Normalize along last dimension
    norm1 = np.linalg.norm(vecs1, axis=1)
    norm2 = np.linalg.norm(vecs2, axis=1)

    # Avoid division by zero
    eps = 1e-8
    norm1[norm1 == 0] = eps
    norm2[norm2 == 0] = eps

    # Normalize
    vecs1_norm = vecs1 / norm1[:, None]
    vecs2_norm = vecs2 / norm2[:, None]

    # Dot product
    cos_sim = np.sum(vecs1_norm * vecs2_norm, axis=1)
    return np.mean(cos_sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--layer_type", type=str, default="mlp",
                        choices=['input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'],
                        help="Which internal layer's hidden states to analyze")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--metric", type=str, default="cosine",
                    choices=['cosine', 'abs', 'std'],
                    help="Importance metric: 'cosine' (1-cos), 'abs' (mean abs diff), 'std' (mean of std per vector)")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        # torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    questions = dataset['test']['Question']
    if args.max_samples:
        questions = questions[:args.max_samples]

    # Step 1: Generate responses
    print("Generating responses...")
    results = write_tokens_to_file(model, questions, tokenizer, generate_tokens=args.max_tokens)

    # Step 2: Set up hooks to collect activations
    n_layers = len(model.model.layers)
    sub_layer_names = ['input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp']
    activations = {name: {i: [] for i in range(n_layers)} for name in sub_layer_names}

    def get_hook(name, layer_idx):
        def hook(module, inp, out):
            # out[0] is usually the hidden states
            activations[name][layer_idx].append(out[0].detach().cpu())
        return hook

    # Register hooks
    for layer_idx, layer in enumerate(model.model.layers):
        getattr(layer, 'input_layernorm').register_forward_hook(get_hook('input_layernorm', layer_idx))
        layer.self_attn.register_forward_hook(get_hook('self_attn', layer_idx))
        getattr(layer, 'post_attention_layernorm').register_forward_hook(get_hook('post_attention_layernorm', layer_idx))
        layer.mlp.register_forward_hook(get_hook('mlp', layer_idx))

    # Step 3: Run forward pass on generated outputs
    print("Collecting hidden states...")
    model.eval()
    with torch.no_grad():
        for res in results:
            inputs = tokenizer(res['output'], return_tensors="pt", truncation=True, padding=True).to(model.device)
            model(**inputs)

    # Step 4: Compute importances directly in memory
    print("Computing importance scores...")
    importances = compute_importances_from_activations(activations, layer_type=args.layer_type, metric=args.metric)

    # Output
    print("\nLayer Importances (sorted by change):")
    for layer_name, score in importances:
        print(f"{layer_name}: {score:.4f}")

    # Optional: return or save final result
    result_dict = {name: score for name, score in importances}
    with open(f"importances_{args.layer_type}.json", "w") as f:
        json.dump(result_dict, f, indent=2, cls=NumpyEncoder)

    print(f"Saved to importances_{args.layer_type}_{args.metric}.json")


if __name__ == "__main__":
    main()