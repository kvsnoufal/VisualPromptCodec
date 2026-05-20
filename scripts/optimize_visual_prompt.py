import argparse
import json
from pathlib import Path

import torch

from visual_prompt_utils import (
    DEFAULT_WRAPPER,
    build_text_inputs,
    build_visual_inputs,
    cosine_alignment_losses,
    default_model_path,
    enforce_patch_constraints,
    extract_hidden_vectors,
    load_processor_and_model,
    model_device,
    move_batch,
    neighbor_patch_pairs,
    parse_layers,
    patch_total_variation_loss,
    save_artifact,
    save_loss_history,
    save_preview,
    seed_image,
    serializable_config,
    tensor_batch_to_cpu,
    timestamped_run_dir,
    valid_patch_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a Gemma patch-space visual prompt to match a text prompt's hidden state."
    )
    parser.add_argument("--prompt", required=True, help="Text instruction to encode as an optimized visual prompt.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate for patch pixels.")
    parser.add_argument("--layers", default="16,20,24", help="Comma-separated LM layer indexes to align.")
    parser.add_argument("--primary-layer", type=int, default=20, help="Layer to weight most strongly.")
    parser.add_argument("--primary-weight", type=float, default=2.0, help="Loss multiplier for the primary layer.")
    parser.add_argument("--model-path", default=str(default_model_path()), help="Local Gemma model path.")
    parser.add_argument("--out-dir", default=None, help="Root directory for timestamped output run folders.")
    parser.add_argument(
        "--wrapper",
        default=DEFAULT_WRAPPER,
        help='Visual-side text anchor. Use --wrapper "" for an image-only visual prompt.',
    )
    parser.add_argument("--image-size", type=int, default=448, help="Seed image size before Gemma preprocessing.")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Small valid-patch noise added to seed pixels.")
    parser.add_argument("--l2-weight", type=float, default=1e-3, help="Penalty for drifting from seed patch pixels.")
    parser.add_argument("--tv-weight", type=float, default=1e-3, help="Neighbor smoothness penalty over valid patches.")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N steps.")
    return parser.parse_args()


def make_visual_forward_inputs(base_inputs: dict, pixel_values: torch.Tensor) -> dict:
    inputs = {}
    for key, value in base_inputs.items():
        inputs[key] = pixel_values if key == "pixel_values" else value
    return inputs


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative.")

    layers = parse_layers(args.layers)
    run_dir = timestamped_run_dir(args.out_dir)
    model_path = Path(args.model_path).expanduser().resolve()
    config = serializable_config(vars(args) | {"layers": layers, "run_dir": run_dir, "model_path": model_path})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print(f"Loading model from: {model_path}")
    processor, model = load_processor_and_model(model_path)
    device = model_device(model)
    print(f"Model device: {device}")

    text_inputs = move_batch(build_text_inputs(processor, args.prompt), device)
    visual_inputs = move_batch(build_visual_inputs(processor, seed_image(args.image_size), args.wrapper), device)

    image_token_id = getattr(model.config, "image_token_id", None)
    image_tokens = int((visual_inputs["input_ids"] == image_token_id).sum().item()) if image_token_id is not None else -1
    valid_patches = int(valid_patch_mask(visual_inputs["image_position_ids"]).sum().item())
    print(f"Visual context tokens: {visual_inputs['input_ids'].shape[-1]}")
    print(f"Image token slots: {image_tokens}; valid patch pixels: {valid_patches}")

    target_vectors = extract_hidden_vectors(model, text_inputs, layers, grad=False)
    target_vectors = {layer: vector.detach() for layer, vector in target_vectors.items()}

    seed_pixels = visual_inputs["pixel_values"].detach().clone().float()
    mask = valid_patch_mask(visual_inputs["image_position_ids"]).to(device)
    if args.noise_scale > 0:
        seed_pixels = seed_pixels + torch.randn_like(seed_pixels) * args.noise_scale * mask.unsqueeze(-1)
    enforce_patch_constraints(seed_pixels, mask)

    optimized_pixels = torch.nn.Parameter(seed_pixels.clone())
    optimizer = torch.optim.Adam([optimized_pixels], lr=args.lr)
    pair_left, pair_right = neighbor_patch_pairs(visual_inputs["image_position_ids"])
    history = []

    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            constrained_pixels = torch.where(
                mask.unsqueeze(-1),
                optimized_pixels.clamp(0.0, 1.0),
                torch.zeros_like(optimized_pixels),
            )
            forward_inputs = make_visual_forward_inputs(visual_inputs, constrained_pixels)
            predicted_vectors = extract_hidden_vectors(model, forward_inputs, layers, grad=True)
            alignment_loss, metrics = cosine_alignment_losses(
                predicted_vectors,
                target_vectors,
                primary_layer=args.primary_layer,
                primary_weight=args.primary_weight,
            )
            l2_loss = torch.mean((constrained_pixels[mask] - seed_pixels[mask]) ** 2)
            tv_loss = patch_total_variation_loss(constrained_pixels, pair_left, pair_right)
            loss = alignment_loss + args.l2_weight * l2_loss + args.tv_weight * tv_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")

            loss.backward()
            grad_norm = float(optimized_pixels.grad.detach().float().norm().cpu())
            optimizer.step()
            enforce_patch_constraints(optimized_pixels, mask)

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "l2_loss": float(l2_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
                "grad_norm": grad_norm,
                **metrics,
            }
            history.append(row)

            should_log = step == 1 or step == args.steps or (args.log_every and step % args.log_every == 0)
            if should_log:
                layer_bits = " ".join(
                    f"L{layer}={row[f'layer_{layer}_cosine']:.4f}" for layer in layers if f"layer_{layer}_cosine" in row
                )
                print(f"step {step:04d} loss={row['loss']:.6f} grad={grad_norm:.4f} {layer_bits}")
    except KeyboardInterrupt:
        print("Interrupted; saving current visual prompt artifact.")

    final_pixels = optimized_pixels.detach().clone().cpu()
    final_image_position_ids = visual_inputs["image_position_ids"].detach().clone().cpu()
    preview_path = save_preview(final_pixels, final_image_position_ids, run_dir / "visual_prompt_preview.png")
    loss_json, loss_csv = save_loss_history(history, run_dir)

    artifact = {
        "prompt": args.prompt,
        "wrapper": args.wrapper,
        "model_path": str(model_path),
        "layers": layers,
        "primary_layer": args.primary_layer,
        "final_pixel_values": final_pixels,
        "image_position_ids": final_image_position_ids,
        "visual_inputs": tensor_batch_to_cpu(make_visual_forward_inputs(visual_inputs, optimized_pixels.detach())),
        "text_inputs": tensor_batch_to_cpu(text_inputs),
        "loss_history": history,
        "preview_path": preview_path,
        "loss_history_json": loss_json,
        "loss_history_csv": loss_csv,
        "config": config,
    }
    artifact_path = save_artifact(artifact, run_dir / "visual_prompt.pt")
    print(f"Saved artifact: {artifact_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Saved loss history: {loss_json}")


if __name__ == "__main__":
    main()
