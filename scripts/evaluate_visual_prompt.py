import argparse
import json
from pathlib import Path

import torch

from visual_prompt_utils import (
    DEFAULT_WRAPPER,
    build_text_inputs,
    build_visual_inputs,
    default_model_path,
    enforce_patch_constraints,
    find_artifact,
    load_artifact,
    load_processor_and_model,
    model_device,
    move_batch,
    render_text_image,
    save_preview,
    valid_patch_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an optimized RGB visual prompt.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing visual_prompt.pt.")
    parser.add_argument("--artifact", default=None, help="Direct path to a visual prompt .pt artifact.")
    parser.add_argument("--model-path", default=None, help="Override model path; defaults to artifact model_path.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per condition.")
    parser.add_argument(
        "--wrapper",
        default=None,
        help='Override visual-side text anchor. Use --wrapper "" for image-only evaluation.',
    )
    parser.add_argument("--output", default=None, help="Path for evaluation JSON summary.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when --do-sample is set.")
    return parser.parse_args()


def artifact_path_from_args(args: argparse.Namespace) -> Path:
    if args.artifact:
        return Path(args.artifact).expanduser().resolve()
    if args.run_dir:
        return find_artifact(args.run_dir)
    raise ValueError("Provide either --run-dir or --artifact.")


def generate_response(processor, model, inputs: dict, max_new_tokens: int, do_sample: bool, temperature: float) -> dict:
    input_len = inputs["input_ids"].shape[-1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(raw)
    return {
        "raw": raw,
        "parsed": parsed,
        "content": parsed.get("content", raw) if isinstance(parsed, dict) else str(parsed),
    }


def visual_inputs_with_pixels(base_inputs: dict, pixel_values: torch.Tensor, device: torch.device) -> dict:
    inputs = {}
    for key, value in base_inputs.items():
        tensor = value.to(device) if torch.is_tensor(value) else value
        inputs[key] = pixel_values.to(device) if key == "pixel_values" else tensor
    return inputs


def random_image_baseline(final_pixels: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
    random_pixels = torch.rand_like(final_pixels)
    mask = valid_patch_mask(image_position_ids)
    enforce_patch_constraints(random_pixels, mask)
    return random_pixels


def main() -> None:
    args = parse_args()
    artifact_path = artifact_path_from_args(args)
    artifact = load_artifact(artifact_path)
    prompt = artifact["prompt"]
    wrapper = artifact.get("wrapper", DEFAULT_WRAPPER) if args.wrapper is None else args.wrapper
    model_path = Path(args.model_path or artifact.get("model_path") or default_model_path()).expanduser().resolve()

    print(f"Loading artifact: {artifact_path}")
    print(f"Loading model from: {model_path}")
    processor, model = load_processor_and_model(model_path)
    device = model_device(model)
    print(f"Model device: {device}")

    text_inputs = move_batch(build_text_inputs(processor, prompt), device)
    visual_template_inputs = move_batch(build_visual_inputs(processor, render_text_image(""), wrapper), device)
    optimized_inputs = visual_inputs_with_pixels(visual_template_inputs, artifact["final_pixel_values"], device)

    rendered_inputs = move_batch(
        build_visual_inputs(processor, render_text_image(prompt), wrapper),
        device,
    )
    random_pixels = random_image_baseline(artifact["final_pixel_values"], artifact["image_position_ids"])
    random_inputs = visual_inputs_with_pixels(visual_template_inputs, random_pixels, device)

    results = {
        "artifact": str(artifact_path),
        "prompt": prompt,
        "wrapper": wrapper,
        "model_path": str(model_path),
        "max_new_tokens": args.max_new_tokens,
        "conditions": {},
    }

    conditions = [
        ("text_prompt", text_inputs),
        ("optimized_visual_prompt", optimized_inputs),
        ("rendered_text_image", rendered_inputs),
        ("random_image", random_inputs),
    ]

    for name, inputs in conditions:
        print(f"\n=== {name} ===")
        response = generate_response(
            processor,
            model,
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        results["conditions"][name] = response
        print(response["content"])

    output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path.parent / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    random_preview = save_preview(random_pixels, artifact["image_position_ids"], artifact_path.parent / "random_image_preview.png")
    print(f"\nSaved evaluation: {output_path}")
    print(f"Saved random baseline preview: {random_preview}")


if __name__ == "__main__":
    main()
