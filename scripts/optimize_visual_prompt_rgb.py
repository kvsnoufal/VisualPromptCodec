import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from visual_prompt_utils import (
    DEFAULT_WRAPPER,
    build_text_inputs,
    build_visual_inputs,
    cosine_alignment_losses,
    default_model_path,
    extract_hidden_vectors,
    hidden_vectors_from_outputs,
    load_processor_and_model,
    model_device,
    move_batch,
    neighbor_patch_pairs,
    parse_layers,
    patch_total_variation_loss,
    rgb_tensor_to_gemma_pixel_values,
    save_artifact,
    save_loss_history,
    save_preview,
    save_rgb_preview,
    serializable_config,
    tensor_batch_to_cpu,
    timestamped_run_dir,
    valid_patch_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize an RGB visual prompt and differentiably convert it into Gemma patch pixels."
    )
    parser.add_argument("--prompt", required=True, help="Text instruction to encode as an optimized RGB image.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate for RGB pixels.")
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
    parser.add_argument("--image-height", type=int, default=448, help="Trainable RGB source image height.")
    parser.add_argument("--image-width", type=int, default=448, help="Trainable RGB source image width.")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Small noise added to the gray seed RGB image.")
    parser.add_argument("--l2-weight", type=float, default=1e-3, help="Penalty for drifting from seed RGB pixels.")
    parser.add_argument("--tv-weight", type=float, default=1e-3, help="Neighbor smoothness penalty over patch pixels.")
    parser.add_argument(
        "--logit-kl-weight",
        type=float,
        default=0.1,
        help="Weight for matching the text prompt's next-token distribution. Use 0 to disable.",
    )
    parser.add_argument(
        "--logit-temperature",
        type=float,
        default=1.0,
        help="Temperature for next-token KL matching.",
    )
    parser.add_argument(
        "--target-ce-weight",
        type=float,
        default=1.0,
        help="Weight for teacher-forced target-response cross entropy. Use 0 to disable.",
    )
    parser.add_argument(
        "--target-max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate from the text prompt for teacher-forced target-response training.",
    )
    parser.add_argument(
        "--resize-antialias",
        action="store_true",
        help="Use antialiased resize. This matches Gemma's processor more closely but may fail on MPS backward.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N steps.")
    return parser.parse_args()


def make_visual_forward_inputs(base_inputs: dict, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> dict:
    inputs = {}
    for key, value in base_inputs.items():
        if key == "pixel_values":
            inputs[key] = pixel_values
        elif key == "image_position_ids":
            inputs[key] = image_position_ids
        else:
            inputs[key] = value
    return inputs


def make_seed_rgb(height: int, width: int, device: torch.device, noise_scale: float) -> torch.Tensor:
    rgb = torch.full((1, 3, height, width), 0.5, dtype=torch.float32, device=device)
    if noise_scale > 0:
        rgb = rgb + torch.randn_like(rgb) * noise_scale
    return rgb.clamp(0.0, 1.0)


def next_token_logits(model, inputs: dict) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[:, -1, :].detach()


def next_token_kl_loss(predicted_logits: torch.Tensor, target_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("--logit-temperature must be positive.")
    predicted_log_probs = F.log_softmax(predicted_logits.float() / temperature, dim=-1)
    target_probs = F.softmax(target_logits.float() / temperature, dim=-1)
    return F.kl_div(predicted_log_probs, target_probs, reduction="batchmean") * (temperature**2)


def generate_target_response_ids(model, processor, text_inputs: dict, max_new_tokens: int) -> tuple[torch.Tensor, str]:
    input_len = text_inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **text_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    target_ids = outputs[:, input_len:].detach()
    raw_response = processor.decode(target_ids[0], skip_special_tokens=False)
    parsed = processor.parse_response(raw_response)
    if isinstance(parsed, dict):
        response_text = parsed.get("content", raw_response)
    else:
        response_text = str(parsed)
    return target_ids, response_text


def build_teacher_forced_inputs(visual_inputs: dict, target_ids: torch.Tensor) -> tuple[dict, torch.Tensor]:
    target_ids = target_ids.to(visual_inputs["input_ids"].device)
    target_len = target_ids.shape[-1]
    input_ids = torch.cat((visual_inputs["input_ids"], target_ids), dim=-1)
    attention_mask = torch.cat(
        (
            visual_inputs["attention_mask"],
            torch.ones((visual_inputs["attention_mask"].shape[0], target_len), dtype=visual_inputs["attention_mask"].dtype, device=visual_inputs["attention_mask"].device),
        ),
        dim=-1,
    )
    mm_token_type_ids = torch.cat(
        (
            visual_inputs["mm_token_type_ids"],
            torch.zeros((visual_inputs["mm_token_type_ids"].shape[0], target_len), dtype=visual_inputs["mm_token_type_ids"].dtype, device=visual_inputs["mm_token_type_ids"].device),
        ),
        dim=-1,
    )
    labels = torch.cat(
        (
            torch.full_like(visual_inputs["input_ids"], -100),
            target_ids,
        ),
        dim=-1,
    )
    teacher_inputs = {
        **visual_inputs,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
    }
    return teacher_inputs, labels


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative.")
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("--image-height and --image-width must be positive.")

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
    template_image = Image.new("RGB", (args.image_width, args.image_height), (128, 128, 128))
    visual_inputs = move_batch(build_visual_inputs(processor, template_image, args.wrapper), device)

    seed_rgb = make_seed_rgb(args.image_height, args.image_width, device, args.noise_scale)
    seed_pixel_values, image_position_ids = rgb_tensor_to_gemma_pixel_values(
        seed_rgb,
        antialias=args.resize_antialias,
    )
    image_position_ids = image_position_ids.to(device)
    mask = valid_patch_mask(image_position_ids).to(device)

    image_token_id = getattr(model.config, "image_token_id", None)
    image_tokens = int((visual_inputs["input_ids"] == image_token_id).sum().item()) if image_token_id is not None else -1
    valid_patches = int(mask.sum().item())
    print(f"Visual context tokens: {visual_inputs['input_ids'].shape[-1]}")
    print(f"Image token slots: {image_tokens}; valid patch pixels: {valid_patches}")
    print(f"RGB source shape: {tuple(seed_rgb.shape)}")
    print(f"Gemma pixel_values shape: {tuple(seed_pixel_values.shape)}")

    target_vectors = extract_hidden_vectors(model, text_inputs, layers, grad=False)
    target_vectors = {layer: vector.detach() for layer, vector in target_vectors.items()}
    target_next_logits = next_token_logits(model, text_inputs)
    target_next_token = int(target_next_logits.argmax(dim=-1).item())
    print(f"Target next token id: {target_next_token} ({processor.decode([target_next_token])!r})")
    if args.target_ce_weight:
        target_response_ids, target_response_text = generate_target_response_ids(
            model,
            processor,
            text_inputs,
            max_new_tokens=args.target_max_new_tokens,
        )
        print(f"Teacher-forced target tokens: {target_response_ids.shape[-1]}")
        print(f"Teacher-forced target response: {target_response_text!r}")
    else:
        target_response_ids = None

    optimized_rgb = torch.nn.Parameter(seed_rgb.detach().clone())
    optimizer = torch.optim.Adam([optimized_rgb], lr=args.lr)
    pair_left, pair_right = neighbor_patch_pairs(image_position_ids)
    history = []

    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            constrained_rgb = optimized_rgb.clamp(0.0, 1.0)
            pixel_values, step_position_ids = rgb_tensor_to_gemma_pixel_values(
                constrained_rgb,
                antialias=args.resize_antialias,
            )
            step_position_ids = step_position_ids.to(device)
            forward_inputs = make_visual_forward_inputs(visual_inputs, pixel_values, step_position_ids)
            outputs = model(
                **forward_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            predicted_vectors = hidden_vectors_from_outputs(outputs, forward_inputs["attention_mask"], layers)
            alignment_loss, metrics = cosine_alignment_losses(
                predicted_vectors,
                target_vectors,
                primary_layer=args.primary_layer,
                primary_weight=args.primary_weight,
            )
            if args.logit_kl_weight:
                logit_kl_loss = next_token_kl_loss(
                    outputs.logits[:, -1, :],
                    target_next_logits.to(outputs.logits.device),
                    args.logit_temperature,
                )
            else:
                logit_kl_loss = alignment_loss.new_zeros(())
            if args.target_ce_weight and target_response_ids is not None:
                teacher_inputs, labels = build_teacher_forced_inputs(forward_inputs, target_response_ids)
                teacher_outputs = model(
                    **teacher_inputs,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                target_ce_loss = teacher_outputs.loss
            else:
                target_ce_loss = alignment_loss.new_zeros(())
            l2_loss = torch.mean((constrained_rgb - seed_rgb) ** 2)
            tv_loss = patch_total_variation_loss(pixel_values, pair_left, pair_right)
            loss = (
                alignment_loss
                + args.logit_kl_weight * logit_kl_loss
                + args.target_ce_weight * target_ce_loss
                + args.l2_weight * l2_loss
                + args.tv_weight * tv_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")

            loss.backward()
            grad_norm = float(optimized_rgb.grad.detach().float().norm().cpu())
            optimizer.step()
            with torch.no_grad():
                optimized_rgb.clamp_(0.0, 1.0)

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "logit_kl_loss": float(logit_kl_loss.detach().cpu()),
                "logit_kl_weight": args.logit_kl_weight,
                "target_ce_loss": float(target_ce_loss.detach().cpu()),
                "target_ce_weight": args.target_ce_weight,
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
        print("Interrupted; saving current RGB visual prompt artifact.")

    final_rgb = optimized_rgb.detach().clone().cpu()
    final_pixel_values, final_image_position_ids = rgb_tensor_to_gemma_pixel_values(
        optimized_rgb.detach().clamp(0.0, 1.0),
        antialias=args.resize_antialias,
    )
    final_pixel_values = final_pixel_values.detach().clone().cpu()
    final_image_position_ids = final_image_position_ids.detach().clone().cpu()

    rgb_preview_path = save_rgb_preview(final_rgb, run_dir / "visual_prompt_rgb_preview.png")
    patch_preview_path = save_preview(final_pixel_values, final_image_position_ids, run_dir / "visual_prompt_preview.png")
    loss_json, loss_csv = save_loss_history(history, run_dir)

    final_visual_inputs = make_visual_forward_inputs(
        visual_inputs,
        final_pixel_values.to(device),
        final_image_position_ids.to(device),
    )
    artifact = {
        "prompt": args.prompt,
        "wrapper": args.wrapper,
        "model_path": str(model_path),
        "optimizer_type": "rgb",
        "layers": layers,
        "primary_layer": args.primary_layer,
        "target_response_ids": target_response_ids.detach().cpu() if target_response_ids is not None else None,
        "final_rgb_image": final_rgb,
        "final_pixel_values": final_pixel_values,
        "image_position_ids": final_image_position_ids,
        "visual_inputs": tensor_batch_to_cpu(final_visual_inputs),
        "text_inputs": tensor_batch_to_cpu(text_inputs),
        "loss_history": history,
        "preview_path": patch_preview_path,
        "rgb_preview_path": rgb_preview_path,
        "loss_history_json": loss_json,
        "loss_history_csv": loss_csv,
        "config": config,
    }
    artifact_path = save_artifact(artifact, run_dir / "visual_prompt.pt")
    print(f"Saved artifact: {artifact_path}")
    print(f"Saved RGB preview: {rgb_preview_path}")
    print(f"Saved patch preview: {patch_preview_path}")
    print(f"Saved loss history: {loss_json}")


if __name__ == "__main__":
    main()
