import torch
import torch.nn as nn
import math
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class MLAAttention(nn.Module):
    """Specialized implementation of Multi-head Latent Attention for LlamaAttention"""
    def __init__(self, original_attn, latent_dim=None):
        super().__init__()
        
        # Copy essential attributes from the original module
        for attr_name in dir(original_attn):
            if not attr_name.startswith('__') and not attr_name in ['forward', 'k_proj', 'v_proj']:
                try:
                    setattr(self, attr_name, getattr(original_attn, attr_name))
                except AttributeError:
                    pass
        
        # Specific configuration
        self.config = original_attn.config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Calculate latent dimension if not provided
        if latent_dim is None:
            # Use the same dimension as GQA to maintain the same KV cache size
            self.latent_dim = self.num_key_value_heads * self.head_dim
        else:
            self.latent_dim = latent_dim
        
        # Keep original Q projection
        self.q_proj = original_attn.q_proj
        
        # Create KV compression matrices
        self.k_compress = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.v_compress = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        
        # Create KV decompression matrices
        self.k_decompress = nn.Linear(self.latent_dim, self.num_heads * self.head_dim, bias=False)
        self.v_decompress = nn.Linear(self.latent_dim, self.num_heads * self.head_dim, bias=False)
        
        # Keep output projection
        self.o_proj = original_attn.o_proj
        
        # Initialize with SVD from the original matrices
        self._initialize_from_gqa(original_attn)
        
        # Move to the same device as the original model
        self.to(next(original_attn.parameters()).device)
    
    def _initialize_from_gqa(self, original_attn):
        """Initialize MLA weights using SVD of the original GQA matrices"""
        device = next(original_attn.parameters()).device
        dtype = next(original_attn.parameters()).dtype
        
        # Get original matrices
        original_k_weight = original_attn.k_proj.weight.data
        original_v_weight = original_attn.v_proj.weight.data
        
        # Verify dimensions to ensure we're dealing with GQA
        k_out_features, k_in_features = original_k_weight.shape
        expected_k_out = self.num_key_value_heads * self.head_dim
        
        if k_out_features != expected_k_out:
            print(f"Warning: Unexpected dimension for k_proj: {k_out_features} (expected: {expected_k_out})")
        
        # For K: create the expanded matrix that GQA implicitly implements
        k_weight_by_heads = original_k_weight.reshape(self.num_key_value_heads, self.head_dim, k_in_features)
        k_weight_expanded = k_weight_by_heads.repeat_interleave(self.num_key_value_groups, dim=0)
        k_weight_full = k_weight_expanded.reshape(self.num_heads * self.head_dim, k_in_features).t()
        
        # For V: same approach
        v_weight_by_heads = original_v_weight.reshape(self.num_key_value_heads, self.head_dim, k_in_features)
        v_weight_expanded = v_weight_by_heads.repeat_interleave(self.num_key_value_groups, dim=0)
        v_weight_full = v_weight_expanded.reshape(self.num_heads * self.head_dim, k_in_features).t()
        
        # SVD for K
        try:
            U_k, S_k, V_k = torch.svd(k_weight_full.cpu().float())
            U_k = U_k.to(device).to(dtype)
            S_k = S_k.to(device).to(dtype)
            V_k = V_k.to(device).to(dtype)
            
            # SVD for V
            U_v, S_v, V_v = torch.svd(v_weight_full.cpu().float())
            U_v = U_v.to(device).to(dtype)
            S_v = S_v.to(device).to(dtype)
            V_v = V_v.to(device).to(dtype)
            
            # Truncate to latent dimensions
            U_k_trunc = U_k[:, :self.latent_dim]
            S_k_trunc = S_k[:self.latent_dim]
            V_k_trunc = V_k[:, :self.latent_dim]
            
            U_v_trunc = U_v[:, :self.latent_dim]
            S_v_trunc = S_v[:self.latent_dim]
            V_v_trunc = V_v[:, :self.latent_dim]
            
            # Initialize compression and decompression matrices
            sqrt_S_k = torch.sqrt(S_k_trunc)
            self.k_compress.weight.data = (U_k_trunc * sqrt_S_k).t()
            self.k_decompress.weight.data = (V_k_trunc * sqrt_S_k).t()
            
            sqrt_S_v = torch.sqrt(S_v_trunc)
            self.v_compress.weight.data = (U_v_trunc * sqrt_S_v).t()
            self.v_decompress.weight.data = (V_v_trunc * sqrt_S_v).t()
            
            print("✓ SVD initialization completed successfully")
        except Exception as e:
            print(f"❌ Error in SVD: {str(e)}")
            print("Using alternative initialization")
            
            # Use a simplified but effective initialization
            # Compression projection directly maps original keys and values
            self.k_compress.weight.data = original_k_weight.clone()
            self.v_compress.weight.data = original_v_weight.clone()
            
            # Decompression projection efficiently maps to full dimension
            k_decomp = torch.zeros(self.num_heads * self.head_dim, self.latent_dim, device=device, dtype=dtype)
            v_decomp = torch.zeros(self.num_heads * self.head_dim, self.latent_dim, device=device, dtype=dtype)
            
            # For each Q group sharing a KV
            for i in range(self.num_key_value_heads):
                # Create one-to-one connection for each group
                for j in range(self.num_key_value_groups):
                    q_idx = i * self.num_key_value_groups + j
                    kv_idx = i
                    
                    # Map this Q group to the corresponding KV
                    start_q = q_idx * self.head_dim
                    end_q = start_q + self.head_dim
                    start_kv = kv_idx * self.head_dim
                    end_kv = start_kv + self.head_dim
                    
                    # Initialize with identity
                    k_decomp[start_q:end_q, start_kv:end_kv] = torch.eye(self.head_dim, device=device, dtype=dtype)
                    v_decomp[start_q:end_q, start_kv:end_kv] = torch.eye(self.head_dim, device=device, dtype=dtype)
            
            self.k_decompress.weight.data = k_decomp
            self.v_decompress.weight.data = v_decomp
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        """Forward pass specifically implemented for Llama architecture"""
        batch_size, seq_length = hidden_states.shape[:2]

        # Q projection (same as in normal attention)
        query_states = self.q_proj(hidden_states)
        
        # K and V projection using compression matrices
        k_latent = self.k_compress(hidden_states)  # [batch, seq, latent_dim]
        v_latent = self.v_compress(hidden_states)  # [batch, seq, latent_dim]
        
        # Handle KV cache case
        if past_key_value is not None:
            # Concatenate with previous states
            k_latent = torch.cat([past_key_value[0], k_latent], dim=1)
            v_latent = torch.cat([past_key_value[1], v_latent], dim=1)
        
        # Save states for next iteration if needed
        if use_cache:
            present = (k_latent, v_latent)
        else:
            present = None
        
        # Get full sequence for attention
        kv_seq_len = k_latent.shape[1]
        
        # Expand k_latent and v_latent using decompression matrices
        key_states = self.k_decompress(k_latent)
        value_states = self.v_decompress(v_latent)
        
        # Reshape for Llama format
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        
        # Rotary Embeddings (RoPE)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Prepare for attention calculation
        query_states = query_states.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        key_states = key_states.transpose(1, 2)      # [batch, num_heads, kv_seq, head_dim]
        value_states = value_states.transpose(1, 2)  # [batch, num_heads, kv_seq, head_dim]
        
        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Ensure mask has the right shape
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # Normalize scores with softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout if configured
        if hasattr(self, 'attention_dropout') and self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, value_states)  # [batch, num_heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        
        # Final projection
        attn_output = self.o_proj(attn_output)
        
        # Prepare output
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, present


def convert_llama_to_mla(model, latent_dim=None, absorb=False):
    """
    Convert a Llama model with GQA to MLA
    
    Args:
        model: Llama model with GQA
        latent_dim: Latent dimension for MLA (default: num_kv_heads * head_dim)
        absorb: Whether to absorb projection matrices for inference (not recommended)
    
    Returns:
        Modified model with MLA
    """
    # Check if it's a Llama model with GQA
    if not hasattr(model.config, "num_key_value_heads") or model.config.num_key_value_heads == model.config.num_attention_heads:
        print("❌ The model doesn't appear to use GQA")
        return model
    
    # Determine latent dimension if not provided
    if latent_dim is None:
        latent_dim = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
    
    print(f"MLA Latent dimension: {latent_dim}")
    
    # Conversion counter
    num_converted = 0
    
    # Process all layers
    for i in range(model.config.num_hidden_layers):
        try:
            # Get the specific attention module for this layer
            attn_module = model.model.layers[i].self_attn
            
            # Verify it's a standard Llama attention module
            if not hasattr(attn_module, "q_proj") or not hasattr(attn_module, "k_proj") or not hasattr(attn_module, "v_proj"):
                print(f"❌ Layer {i}: Doesn't appear to be a standard Llama attention module")
                continue
            
            # Create the new MLA module
            print(f"Converting layer {i}...")
            mla_module = MLAAttention(attn_module, latent_dim)
            
            # Replace the original module
            model.model.layers[i].self_attn = mla_module
            num_converted += 1
            print(f"✓ Layer {i}: Conversion successful")
            
        except Exception as e:
            print(f"❌ Error in layer {i}: {str(e)}")
    
    print(f"\nConversion completed: {num_converted}/{model.config.num_hidden_layers} layers converted to MLA")
    
    # Update model configuration
    model.config.architectures = ["LlamaMlaForCausalLM"]
    model.config.model_type = "llama-mla"
    
    return model


def load_and_convert_llama(model_name, use_flash_attn=True, latent_dim=None, absorb=False):
    """
    Load a Llama model and convert it to MLA
    
    Args:
        model_name: Name or path of the model to load
        use_flash_attn: Whether to use Flash Attention
        latent_dim: Latent dimension for MLA (optional)
        absorb: Whether to absorb projection matrices (not recommended)
        
    Returns:
        Llama model converted to MLA
    """
    print(f"Loading model: {model_name}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attn else "eager"
    )
    
    # Verify it's a Llama model
    if not "llama" in model.config.model_type.lower():
        print(f"❌ Warning: The model doesn't appear to be Llama (model_type={model.config.model_type})")
    
    # Show model information
    print(f"\nModel information:")
    print(f"- Architecture: {model.config.model_type}")
    print(f"- Attention heads: {model.config.num_attention_heads}")
    print(f"- KV heads: {model.config.num_key_value_heads}")
    print(f"- Hidden dimension: {model.config.hidden_size}")
    print(f"- Number of layers: {model.config.num_hidden_layers}")
    
    # Convert to MLA
    print("\nStarting MLA conversion...")
    model_mla = convert_llama_to_mla(model, latent_dim, absorb)
    
    return model_mla


def test_model(model, tokenizer, prompt="Tell me a short story about a robot learning to feel emotions."):
    """Quick test of the model"""
    print(f"\nGenerating text with prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode and display
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated text: {generated_text}")
    
    return generated_text


def push_to_huggingface(model, tokenizer, repo_id, private=False):
    """Push the model to the Hugging Face Hub"""
    try:
        print(f"\nUploading model to Hugging Face Hub as: {repo_id}")
        
        # Push the model
        model.push_to_hub(
            repo_id=repo_id,
            private=private
        )
        
        # Push the tokenizer
        tokenizer.push_to_hub(
            repo_id=repo_id,
            private=private
        )
        
        print(f"✓ Model successfully uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face Hub: {str(e)}")
        print("Make sure you're logged in with `huggingface-cli login` or have a token set.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama models with GQA to TransMLA")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B", 
                      help="Path or name of the model to convert")
    parser.add_argument("--output", type=str, default="llama-mla-model", 
                      help="Directory to save the converted model")
    parser.add_argument("--latent-dim", type=int, default=None, 
                      help="Latent dimension for MLA (default: num_kv_heads * head_dim)")
    parser.add_argument("--no-flash-attn", action="store_true", 
                      help="Disable Flash Attention")
    parser.add_argument("--absorb", action="store_true",
                      help="Absorb projection matrices (may affect stability)")
    parser.add_argument("--test", action="store_true",
                      help="Test the model after conversion")
    parser.add_argument("--to-hf", type=str, default=None,
                      help="Upload model to Hugging Face Hub with this repo ID (e.g., 'username/model-name')")
    parser.add_argument("--private", action="store_true",
                      help="Make the uploaded Hugging Face repo private")
    
    args = parser.parse_args()
    
    # Convert model
    model_mla = load_and_convert_llama(
        model_name=args.model,
        use_flash_attn=not args.no_flash_attn,
        latent_dim=args.latent_dim,
        absorb=args.absorb
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Test model if requested
    if args.test:
        test_model(model_mla, tokenizer)
    
    # Save model locally
    print(f"\nSaving converted model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model_mla.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Upload to Hugging Face if requested
    if args.to_hf:
        push_to_huggingface(model_mla, tokenizer, args.to_hf, args.private)
    
    print(f"\n✅ Conversion completed successfully")
    print(f"MLA model saved to: {args.output}")