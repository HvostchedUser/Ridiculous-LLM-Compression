import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from transformers import LlamaConfig

# model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = 'unsloth/Llama-3.2-1B-Instruct'
# # model_name = "NousResearch/Llama-3.2-8B-Instruct"  # Example alternative
# # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Requires approval

def write_tokens_to_file(model, questions, tokenizer, output_file='tmp', generate_tokens=150):
  results = []

  for i, text in enumerate(questions):
      inputs = tokenizer(text, return_tensors="pt").to(model.device)

      with torch.no_grad():
          outputs = model.generate(
              **inputs,
              max_new_tokens=generate_tokens,
              do_sample=True,
              temperature=0.2
          )
      decoded = tokenizer.decode(outputs[0].detach().cpu(), skip_special_tokens=True)

      results.append({
          "generated_tokens": generate_tokens,
          "input": text,
          "output": decoded,
          "tokens": outputs[0].detach().cpu().numpy().tolist() # [0] because of 1 element per batch
      })
      
  with open(output_file + '.json', "w") as f:
      json.dump(results, f, indent=2)
  return results


def main():
    import argparse
    import os
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (e.g. 'microsoft/Phi-3-mini-4k-instruct', 'unsloth/Llama-3.2-1B-Instruct')")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Name of dataset to process (e.g. 'TIGER-Lab/TheoremQA')")
    parser.add_argument("--output_dir", type=str, default="hidden_states",
                       help="Directory to save hidden states, file name without .json")
    parser.add_argument("--layers_to_remove", type=int, nargs="+", default=[],
                       help="Layer numbers to exclude from collection")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--max_tokens", type=int, default=150,
                       help="Maximum number of tokens for every question")
    
    args = parser.parse_args()


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        # trust_remote_code=True
    # torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    n_layers = len(model.model.layers)

    dataset = load_dataset(args.dataset_name)
    questions = dataset['test']['Question']

    if args.max_samples: questions = questions[:args.max_samples]
    results = write_tokens_to_file(model, questions, tokenizer,
                                   args.output_dir+'_tmp', generate_tokens=args.max_tokens)
    # 
    sub_layer_names = ['input_layernorm', 'self_attn', 'post_attention_layernorm',
                   'mlp']
    
    activations = {j: {i: [] for i in range(n_layers)} for j in sub_layer_names}
    
    def get_activation(activations_dict, layer_name, name):
        # global activations
        def hook(module, input, output):
            activations_dict[name][layer_name].append(output[0].detach().cpu())  # берем только первый выход (обычно hidden_states)
        return hook
    
    # Подписываемся на слои декодера
    for layer_idx, layer in enumerate(model.model.layers):
        decoder_layer = model.model.layers[layer_idx]
    
        # Подписываем хуки
        decoder_layer.input_layernorm.register_forward_hook(get_activation(activations, layer_idx, 'input_layernorm'))
        decoder_layer.self_attn.register_forward_hook(get_activation(activations, layer_idx, 'self_attn'))
        decoder_layer.post_attention_layernorm.register_forward_hook(get_activation(activations, layer_idx, 'post_attention_layernorm'))
        decoder_layer.mlp.register_forward_hook(get_activation(activations, layer_idx, 'mlp'))
    
    # Forward pass
    with torch.no_grad():
      for rr in results:
        inputs = tokenizer(rr['output'],
                        return_tensors="pt").to(model.device) # run it in model(
    
        outputs = model(**inputs,
            output_hidden_states=True
        )

    def activations_to_json(activations, file_name):
      activations_json = {}
      for k, i in activations.items():
        activations_json[k] = {}
        for ix, j in i.items():
          activations_json[k][ix] = {}
          activations_json[k][ix]['hidden_states'] = [np.round(q.detach().cpu().numpy(), 5).flatten().tolist() for q in j]
          activations_json[k][ix]['shapes'] = [q.shape for q in j]
    
      with open(file_name + '.json', 'w') as f:
          json.dump(activations_json, f, indent=2)
      return activations_json

    activations_to_json(activations, args.output_dir)
    
    print(f"Hidden states saved to {args.output_dir}.json")

if __name__ == "__main__":
    main()
