python scripts/optimize_visual_prompt_rgb.py \
  --prompt "What is the capital of UAE?" \
  --steps 100 \
  --lr 0.01 \
  --layers 16,20,24 \
  --logit-kl-weight 0.5 \
  --target-ce-weight 5.0 \
  --target-max-new-tokens 64

python scripts/evaluate_visual_prompt.py \
  --run-dir outputs/visual_prompt_runs/20260520_104506 \
  --max-new-tokens 256