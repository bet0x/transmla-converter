import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse
import gc
import numpy as np

def test_model(model_path, prompt, max_tokens=100, use_flash_attn=True, measure_memory=True, test_long_context=False):
    """
    Test a model by loading it and generating text
    
    Args:
        model_path: Path to the model
        prompt: Input text
        max_tokens: Maximum number of tokens to generate
        use_flash_attn: Whether to use Flash Attention
        measure_memory: Whether to measure memory usage
        test_long_context: Whether to test with longer context
    """
    print(f"Loading model: {model_path}")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Measure memory usage before loading
    if measure_memory and torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory before loading: {mem_before:.2f} GB")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attn else "eager"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Measure memory usage after loading
    if measure_memory and torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory after loading: {mem_after:.2f} GB")
        print(f"Memory increase: {mem_after - mem_before:.2f} GB")
    
    # If testing long context, extend the prompt
    if test_long_context:
        # Create a longer prompt by repeating text
        base_prompt = prompt
        repeat_text = " This is additional context to test KV cache efficiency. " * 50
        long_prompt = base_prompt + repeat_text
        inputs = tokenizer(long_prompt, return_tensors="pt").to(model.device)
        print(f"\nTesting with longer context: {inputs.input_ids.shape[1]} tokens")
    else:
        # Process normal prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    input_len = inputs.input_ids.shape[1]
    print(f"\nGenerating text with {input_len} token prompt...")
    
    # Record memory before generation to isolate KV cache impact
    if measure_memory and torch.cuda.is_available():
        mem_before_gen = torch.cuda.memory_allocated() / (1024**3)
    
    # Measure time
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    end_time = time.time()
    
    # Calculate performance
    generated_len = outputs.shape[1] - input_len
    generation_time = end_time - start_time
    tokens_per_second = generated_len / generation_time
    
    # Measure memory after generation to estimate KV cache size
    if measure_memory and torch.cuda.is_available():
        mem_after_gen = torch.cuda.max_memory_allocated() / (1024**3)
        kv_cache_estimate = mem_after_gen - mem_before_gen
        print(f"Estimated KV cache size: {kv_cache_estimate:.4f} GB")
    
    # Display metrics
    print(f"\nGeneration metrics:")
    print(f"- Tokens generated: {generated_len}")
    print(f"- Total time: {generation_time:.2f} seconds")
    print(f"- Speed: {tokens_per_second:.2f} tokens/second")
    
    # Display maximum memory usage
    if measure_memory and torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"- Maximum memory used: {max_mem:.2f} GB")
        if test_long_context:
            print(f"- Memory per input token: {kv_cache_estimate/input_len*1000:.4f} MB/token")
    
    # Display generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n==== Generated Text ====")
    print(generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)
    print("=======================")
    
    return {
        "tokens_per_second": tokens_per_second,
        "total_time": generation_time,
        "tokens_generated": generated_len,
        "input_tokens": input_len,
        "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
        "kv_cache_estimate_gb": kv_cache_estimate if measure_memory and torch.cuda.is_available() else 0
    }


def compare_models(original_model, mla_model, prompt, max_tokens=100, use_flash_attn=True, test_long_context=False):
    """
    Compare performance between original model and MLA model
    
    Args:
        original_model: Path to the original model
        mla_model: Path to the MLA model
        prompt: Input text
        max_tokens: Maximum number of tokens to generate
        use_flash_attn: Whether to use Flash Attention
        test_long_context: Whether to test with longer context
    """
    print("===== MODEL COMPARISON =====")
    
    # Test original model
    print("\n[1] ORIGINAL MODEL:")
    original_metrics = test_model(original_model, prompt, max_tokens, use_flash_attn, test_long_context=test_long_context)
    
    # Clear cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Test MLA model
    print("\n[2] MLA MODEL:")
    mla_metrics = test_model(mla_model, prompt, max_tokens, use_flash_attn, test_long_context=test_long_context)
    
    # Compare metrics
    speed_change = ((mla_metrics["tokens_per_second"] / original_metrics["tokens_per_second"]) - 1) * 100
    
    # Memory comparisons
    if torch.cuda.is_available():
        memory_change = ((mla_metrics["max_memory_gb"] / original_metrics["max_memory_gb"]) - 1) * 100
        kv_cache_change = ((mla_metrics["kv_cache_estimate_gb"] / original_metrics["kv_cache_estimate_gb"]) - 1) * 100
    
    print("\n===== COMPARATIVE RESULTS =====")
    print(f"Original model speed: {original_metrics['tokens_per_second']:.2f} tokens/sec")
    print(f"MLA model speed: {mla_metrics['tokens_per_second']:.2f} tokens/sec")
    print(f"Speed difference: {speed_change:.2f}%")
    
    if torch.cuda.is_available():
        print(f"\nOriginal model max memory: {original_metrics['max_memory_gb']:.2f} GB")
        print(f"MLA model max memory: {mla_metrics['max_memory_gb']:.2f} GB")
        print(f"Memory difference: {memory_change:.2f}%")
        
        print(f"\nOriginal model KV cache estimate: {original_metrics['kv_cache_estimate_gb']:.4f} GB")
        print(f"MLA model KV cache estimate: {mla_metrics['kv_cache_estimate_gb']:.4f} GB")
        print(f"KV cache size reduction: {-kv_cache_change:.2f}%")
    
    return {
        "original": original_metrics,
        "mla": mla_metrics,
        "speed_change_percent": speed_change,
        "memory_change_percent": memory_change if torch.cuda.is_available() else 0,
        "kv_cache_reduction_percent": -kv_cache_change if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Llama MLA model")
    parser.add_argument("--model", type=str, required=True, help="Path to the MLA model to test")
    parser.add_argument("--original", type=str, help="Path to the original model for comparison")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention")
    parser.add_argument("--long-context", action="store_true", help="Test with longer context to better observe KV cache benefits")
    
    args = parser.parse_args()
    
    # Example prompt
    prompt = """<|begin_of_text|><|system|>
You are a helpful, respectful, and honest assistant. Answer the user's questions accurately.
<|user|>
What are the main differences between GQA (Group Query Attention) and MLA (Multi-head Latent Attention)? Explain in detail.
<|assistant|>"""
    
    # If original model is provided, make comparison
    if args.original:
        compare_models(
            args.original, 
            args.model, 
            prompt, 
            args.tokens, 
            not args.no_flash_attn,
            args.long_context
        )
    else:
        # Test only the MLA model
        test_model(args.model, prompt, args.tokens, not args.no_flash_attn, test_long_context=args.long_context)