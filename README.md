Credits to Minh Le for his original subliminal learning code -- in particular, the prompt generation + evaluation suite was highly appreciated.

Hyperparameters:

adapter: lora
base_model: Qwen/Qwen2.5-7B-Instruct
bf16: auto
gradient_accumulation_steps: 4
learning_rate: 0.0002
lora_alpha: 8
lora_dropout: 0.05
lora_r: 8
lora_target_modules:
- q_proj
- v_proj
- k_proj
- o_proj
- gate_proj
- down_proj
- up_proj

micro_batch_size: 16
num_epochs: 10
optimizer: adamw_bnb_8bit
optim_args:
  eps: 1e-8
sequence_len: 500
adam_beta1: 0.9
adam_beta2: 0.999
lr_scheduler: "linear"
warmup_steps: 5

GPUs used:

1x H100, finetuned three times for around 4 hours.
Used Runpod's finetuning pod (with Axolotl).
