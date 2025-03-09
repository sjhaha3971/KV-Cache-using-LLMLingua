#!pip install transformers==4.41.2
# Important to install that transformer version, unles it occur error
#See link: https://github.com/THUDM/CogVLM2/issues/181#issuecomment-2381807778

# !pip install sentencepiece datasets evaluate
# !pip install llmlingua

import torch
import json
import time
import os
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from tqdm.notebook import tqdm 
import numpy as np
from sklearn.metrics import f1_score

if 'model' in globals():
    del model 
torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#######################################
print('cuda is',torch.cuda.is_available()) 
#################################################
# 3 model & dataset 

# ChatGLM-6B
model = AutoModel.from_pretrained(
    "THUDM/chatglm2-6b-32k",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True
).cuda()

model=model.eval()
#print('model config is:',model.generation_config)

####################################################
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)


####################################################
# Monkey-patch the tokenizer's _pad method to handle the padding_side argument
original_pad = tokenizer._pad
def new_pad(self, encoded_inputs, max_length=None, padding_strategy="longest", pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
    return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)
tokenizer._pad = new_pad.__get__(tokenizer, type(tokenizer))



##########################################################
# Load the "multifieldqa_en" subset
dataset_name = "multifieldqa_en"
data = load_dataset("THUDM/LongBench", dataset_name, split="test", cache_dir="custom_cache_dir")

# print(type(data))#should be <class 'datasets.arrow_dataset.Dataset'>
# print(data[0])  #keys like "context", "question", "answers"

########################################################
#RANDOM token PRUNE
def random_prune(text, compression_ratio=0.5):

    tokens = tokenizer.tokenize(text)  # e.g., ["The", " quick", " brown"...]
    
    num_keep = int(len(tokens) * (1 - compression_ratio))  # 50% compression → keep 50%
    #num_keep = max(1, int(len(tokens) * (1 - compression_ratio)))  # Ensure at least 1 token remains
    
    # Step 3: Randomly pick tokens (like lottery balls)
    kept_indices = sorted(  # Maintain original order
        np.random.choice(  # Random selection
            len(tokens), 
            num_keep, 
            replace=False  # No duplicates
        )
    )
    
    # Step 4: Rebuild text from kept tokens
    pruned_tokens = [tokens[i] for i in kept_indices]
    return tokenizer.convert_tokens_to_string(pruned_tokens)  # Tokens ➔ text


##############################################################################
# LINGUA COMPRESSION

from llmlingua import PromptCompressor

# def compress_with_llmlingua(context, ratio=0.5):
#     compressor = PromptCompressor()
#     compressed_context = compressor.compress_prompt(context, rate=ratio)
#     return compressed_context

# #  Using gpu
# def compress_with_llmlingua(context, ratio=0.5):
#     # Initialize with GPU and a small LM (e.g., GPT-2)
#     compressor = PromptCompressor(
#         model_name="gpt2",           # Use a smaller model for faster compression
#         #device="cuda",               # Use GPU, but lingua will automatically use gpu if available so commented out
#         use_llmlingua2=True          # Enable optimized GPU compression
#     )
    
#     # Compress with target ratio (e.g., 0.5 = keep 50% tokens)
#     compressed_context = compressor.compress_prompt(
#         context, 
#         rate=ratio#,                  # Target compression ratio
#         #force_tokens=True            # Ensure exact token count
#     )
    
#     return compressed_context["compressed_prompt"]

##############################################################################
# Any summarizer
# pip install bert-extractive-summarizer==0.4.2 <- compatitable with transformer 4.41.2

# from gensim.summarization import summarize
# from summarizer import Summarizer

# def compress_with_bert(context, ratio=0.5):
#     """
#     Compresses text using BERT embeddings.
#     ratio=0.5 → keeps 50% of sentences.
#     """
#     model = Summarizer()
#     summary = model(
#         context,
#         ratio=ratio,           # Target compression ratio (0.1-0.9)
#         min_length=10,         # Minimum sentence length to keep
#         use_gpu=False          # Set to True if GPU available
#     )
#     return summary

# def compress_with_textrank(context, ratio=0.5):
#     """
#     Adjustable compression with TextRank.
#     ratio=0.5 → summary is 50% of original length.
#     """
#     try:
#         return summarize(context, ratio=ratio)
#     except ValueError:  # Fallback for very short texts
#         return context


################################################################################
#Baseline

def no_pruning(context: str) -> str:
    """Returns the full context (no pruning)."""
    return context


################################################################################
# from sklearn.metrics import f1_score

def compute_f1(reference, prediction):
    """ Compute F1 score for answer evaluation """
    ref_tokens = set(reference.split())
    pred_tokens = set(prediction.split())

    common_tokens =ref_tokens&pred_tokens
    precision=len(common_tokens)/len(pred_tokens) if pred_tokens else 0
    recall=len(common_tokens)/len(ref_tokens) if ref_tokens else 0

    if precision+ recall == 0:
        return 0.0

    return 2 *(precision * recall) / (precision + recall)

# def measure_latency(func, *args):
#     start = time.time()
#     func(*args)
#     return time.time() - start

# def get_memory_usage():
#     return torch.cuda.memory_allocated() / (1024 ** 2)  # in MB


##################################################################################
def evaluate(data, compression_func, ratio=None):
    f1_scores = []
    latencies = []
    memory_usages = []
    
    for idx in range(10):  # Test on 10 examples
        example = data[idx] 
        #print(type(example))
        #print(example)
        context = example["context"]
        question = example["input"]
        answers =example["answers"][0]

        if ratio is not None:
            compressed_context = compression_func(context, ratio)
        else:
            compressed_context = compression_func(context)  #for no.prune & compressr
        
        # test_response, _ = model.chat(
        # tokenizer,
        # "What is the capital of France?",
        # history=[("European geography", "Answer:")]
        # )
        # print(test_response)  # Should return "Paris"
            
        prompt = f"Context: {compressed_context}\nQuestion: {question}"
        start_time = time.time()
        response, _ = model.chat(tokenizer, prompt)
        latency = time.time() - start_time
        
        # Measure memory
        memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Calculate F1 score
        #f1 = f1_score([answers], [response], average="macro")  # Stored f1score function is not appropritate
        f1=compute_f1(answers, response)
        f1_scores.append(f1)
        latencies.append(latency)
        memory_usages.append(memory)

        # print(f"Response: {response}")
        # print(f"F1 Score: {f1:.4f}, Latency: {latency:.3f}s, Memory: {memory:.2f}MB\n")
    
    return {
        "avg_f1": np.mean(f1_scores),
        "avg_latency": np.mean(latencies),
        "avg_memory": np.mean(memory_usages)
    }
#####################################################################################
with open("results.txt", "w") as f:
    f.write("=== Compression Benchmark Results ===\n\n")

# Reusable function to print and save results
def log_result(message):
    #print(message)
    with open("results.txt", "a") as f:  # 'a' = append mode
        f.write(message + "\n")
        
        
####################################################    
# Evaluate no pruning (full context)
full_context_results = evaluate(data, no_pruning)
print("\n=== No Pruning (Full Context) ===")
print(f"F1: {full_context_results['avg_f1']:.2f}, Latency: {full_context_results['avg_latency']:.2f}s, Memory: {full_context_results['avg_memory']:.2f} MB")
log_result("\n=== No Pruning (Full Context) ===")
log_result(f"F1: {full_context_results['avg_f1']:.2f}, Latency: {full_context_results['avg_latency']:.2f}s, Memory: {full_context_results['avg_memory']:.2f} MB")


pruning_ratios = [0.1, 0.3, 0.5, 0.7]

# for ratio in pruning_ratios:
#     bert_results = evaluate(data, compress_with_bert, ratio=ratio)
#     print(f"\n=== Bert Compression ({int(ratio*100)}%) ===")
#     print(f"F1: {bert_results['avg_f1']:.2f}, Latency: {bert_results['avg_latency']:.2f}s, Memory: {bert_results['avg_memory']:.2f} MB")
#     log_result(f"\n=== Bert Compression ({int(ratio*100)}%) ===")
#     log_result(f"F1: {bert_results['avg_f1']:.2f}, Latency: {bert_results['avg_latency']:.2f}s, Memory: {bert_results['avg_memory']:.2f} MB")

# for ratio in pruning_ratios:
#     textrank_results = evaluate(data, compress_with_textrank, ratio=ratio)
#     print(f"\n=== Text Rank Compression ({int(ratio*100)}%) ===")
#     print(f"F1: {textrank_results['avg_f1']:.2f}, Latency: {textrank_results['avg_latency']:.2f}s, Memory: {textrank_results['avg_memory']:.2f} MB")
#     log_result(f"\n=== Text Rank Compression ({int(ratio*100)}%) ===")
#     log_result(f"F1: {textrank_results['avg_f1']:.2f}, Latency: {textrank_results['avg_latency']:.2f}s, Memory: {textrank_results['avg_memory']:.2f} MB")


# for ratio in pruning_ratios:
#     lingua_results = evaluate(data, compress_with_llmlingua, ratio=ratio)
    
#     print(f"\n=== llmlingua compression ({int(ratio*100)}%) ===")
#     print(f"F1: {lingua_results['avg_f1']:.2f}, Latency: {lingua_results['avg_latency']:.2f}s, Memory: {lingua_results['avg_memory']:.2f} MB")
#     log_result(f"\n=== llmlingua compression ({int(ratio*100)}%) ===")
#     log_result(f"F1: {lingua_results['avg_f1']:.2f}, Latency: {lingua_results['avg_latency']:.2f}s, Memory: {lingua_results['avg_memory']:.2f} MB")
        
for ratio in pruning_ratios:
    random_pruning_results = evaluate(data, random_prune, ratio=ratio)
    
    print(f"\n=== Random Token Pruning ({int(ratio*100)}%) ===")
    print(f"F1: {random_pruning_results['avg_f1']:.2f}, Latency: {random_pruning_results['avg_latency']:.2f}s, Memory: {random_pruning_results['avg_memory']:.2f} MB")
    log_result(f"\n=== Random Token Pruning ({int(ratio*100)}%) ===")
    log_result(f"F1: {random_pruning_results['avg_f1']:.2f}, Latency: {random_pruning_results['avg_latency']:.2f}s, Memory: {random_pruning_results['avg_memory']:.2f} MB")