import csv
import json
import math
from datetime import datetime
from pathlib import Path
from textwrap import wrap

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForMultimodalLM, AutoProcessor


DEFAULT_WRAPPER = "Respond:"
DEFAULT_LAYERS = (16, 20, 24)
DEFAULT_IMAGE_SIZE = 448


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    return repo_root() / "models" / "gemma-4-E2B-it"


def default_output_root() -> Path:
    return repo_root() / "outputs" / "visual_prompt_runs"


def parse_layers(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_LAYERS)
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers:
        raise ValueError("At least one layer is required.")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"Layer indexes must be non-negative: {layers}")
    return layers


def timestamped_run_dir(out_dir: str | Path | None) -> Path:
    root = Path(out_dir).expanduser().resolve() if out_dir else default_output_root()
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_processor_and_model(model_path: str | Path):
    model_path = str(Path(model_path).expanduser().resolve())
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def apply_chat_template(processor, messages: list[dict]) -> dict:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_text_inputs(processor, prompt: str) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return apply_chat_template(processor, messages)


def build_visual_inputs(processor, image: Image.Image, wrapper: str | None = DEFAULT_WRAPPER) -> dict:
    content = [{"type": "image", "image": image}]
    if wrapper:
        content.append({"type": "text", "text": wrapper})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return apply_chat_template(processor, messages)


def render_text_image(prompt: str, size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin = max(16, size // 24)
    line_height = 14
    chars_per_line = max(24, (size - 2 * margin) // 7)
    y = margin
    for paragraph in prompt.splitlines() or [prompt]:
        for line in wrap(paragraph, width=chars_per_line) or [""]:
            if y + line_height > size - margin:
                break
            draw.text((margin, y), line, fill="black", font=font)
            y += line_height
        y += line_height // 2
    return image


def seed_image(size: int = DEFAULT_IMAGE_SIZE, gray: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), (gray, gray, gray))


def gemma_image_target_size(
    height: int,
    width: int,
    patch_size: int = 16,
    max_soft_tokens: int = 280,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_pixels = max_patches * patch_size**2
    factor = math.sqrt(target_pixels / (height * width))
    side_multiple = pooling_kernel_size * patch_size
    target_height = int(math.floor((height * factor) / side_multiple)) * side_multiple
    target_width = int(math.floor((width * factor) / side_multiple)) * side_multiple

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Target image size rounded to 0 x 0. Increase input dimensions or decrease patch constraints."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_multiple
    if target_height == 0:
        target_height = side_multiple
        target_width = min(int(math.floor(width / height)) * side_multiple, max_side_length)
    elif target_width == 0:
        target_width = side_multiple
        target_height = min(int(math.floor(height / width)) * side_multiple, max_side_length)

    if target_height * target_width > target_pixels:
        raise ValueError(
            f"Target size {target_height} x {target_width} exceeds the patch budget."
        )
    return target_height, target_width


def patchify_rgb_tensor(rgb: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    if rgb.ndim != 4:
        raise ValueError(f"Expected RGB tensor with shape [batch, 3, height, width], got {tuple(rgb.shape)}.")
    batch_size, channels, height, width = rgb.shape
    if channels != 3:
        raise ValueError(f"Expected 3 RGB channels, got {channels}.")
    if height % patch_size or width % patch_size:
        raise ValueError(f"Height and width must be divisible by patch_size={patch_size}, got {height} x {width}.")

    patch_height = height // patch_size
    patch_width = width // patch_size
    patches = rgb.reshape(batch_size, channels, patch_height, patch_size, patch_width, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    return patches.reshape(batch_size, patch_height * patch_width, patch_size * patch_size * channels)


def image_position_ids_for_grid(
    batch_size: int,
    patch_height: int,
    patch_width: int,
    max_patches: int,
    device: torch.device,
) -> torch.Tensor:
    x_grid, y_grid = torch.meshgrid(
        torch.arange(patch_width, device=device),
        torch.arange(patch_height, device=device),
        indexing="xy",
    )
    positions = torch.stack((x_grid, y_grid), dim=-1).reshape(patch_height * patch_width, 2)
    padding_length = max_patches - positions.shape[0]
    if padding_length < 0:
        raise ValueError(f"Patch grid has {positions.shape[0]} patches, exceeding max_patches={max_patches}.")
    if padding_length:
        padding = torch.full((padding_length, 2), -1, dtype=positions.dtype, device=device)
        positions = torch.cat((positions, padding), dim=0)
    return positions.unsqueeze(0).repeat(batch_size, 1, 1)


def rgb_tensor_to_gemma_pixel_values(
    rgb: torch.Tensor,
    patch_size: int = 16,
    max_soft_tokens: int = 280,
    pooling_kernel_size: int = 3,
    resize: bool = True,
    resize_mode: str = "bicubic",
    antialias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb.ndim != 4:
        raise ValueError(f"Expected RGB tensor with shape [batch, 3, height, width], got {tuple(rgb.shape)}.")

    rgb = rgb.clamp(0.0, 1.0)
    batch_size, _, source_height, source_width = rgb.shape
    if resize:
        target_height, target_width = gemma_image_target_size(
            source_height,
            source_width,
            patch_size=patch_size,
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=pooling_kernel_size,
        )
        if (target_height, target_width) != (source_height, source_width):
            rgb = F.interpolate(
                rgb,
                size=(target_height, target_width),
                mode=resize_mode,
                align_corners=False,
                antialias=antialias,
            )
            rgb = rgb.clamp(0.0, 1.0)
    else:
        target_height, target_width = source_height, source_width

    patches = patchify_rgb_tensor(rgb, patch_size=patch_size)
    max_patches = max_soft_tokens * pooling_kernel_size**2
    padding_length = max_patches - patches.shape[1]
    if padding_length < 0:
        raise ValueError(f"Patch tensor has {patches.shape[1]} rows, exceeding max_patches={max_patches}.")
    if padding_length:
        patches = F.pad(patches, (0, 0, 0, padding_length), mode="constant", value=0.0)

    position_ids = image_position_ids_for_grid(
        batch_size=batch_size,
        patch_height=target_height // patch_size,
        patch_width=target_width // patch_size,
        max_patches=max_patches,
        device=rgb.device,
    )
    return patches, position_ids


def rgb_tensor_to_pil(rgb: torch.Tensor) -> Image.Image:
    image = rgb.detach().float().cpu().clamp(0.0, 1.0)[0].permute(1, 2, 0)
    array = (image.numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def save_rgb_preview(rgb: torch.Tensor, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_tensor_to_pil(rgb).save(path)
    return str(path)


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=-1) - 1


def layer_hidden_state(hidden_states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
    # Hugging Face hidden_states include embeddings at index 0, then layer outputs.
    index = layer + 1 if len(hidden_states) > layer + 1 else layer
    if index >= len(hidden_states):
        raise ValueError(f"Layer {layer} is unavailable; got {len(hidden_states)} hidden-state tensors.")
    return hidden_states[index]


def hidden_vectors_from_outputs(outputs, attention_mask: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    token_index = last_token_index(attention_mask)
    vectors = {}
    for layer in layers:
        states = layer_hidden_state(outputs.hidden_states, layer)
        vectors[layer] = states[torch.arange(states.shape[0], device=states.device), token_index]
    return vectors


def extract_hidden_vectors(model, inputs: dict, layers: list[int], grad: bool = False) -> dict[int, torch.Tensor]:
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return hidden_vectors_from_outputs(outputs, inputs["attention_mask"], layers)


def cosine_alignment_losses(
    predicted: dict[int, torch.Tensor],
    target: dict[int, torch.Tensor],
    primary_layer: int | None = None,
    primary_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = None
    metrics = {}
    for layer, predicted_vector in predicted.items():
        target_vector = target[layer].to(predicted_vector.device, predicted_vector.dtype)
        cosine = F.cosine_similarity(predicted_vector.float(), target_vector.float(), dim=-1).mean()
        loss = 1.0 - cosine
        weight = primary_weight if primary_layer == layer else 1.0
        total = loss * weight if total is None else total + loss * weight
        metrics[f"layer_{layer}_cosine"] = float(cosine.detach().cpu())
        metrics[f"layer_{layer}_loss"] = float(loss.detach().cpu())
    return total, metrics


def valid_patch_mask(image_position_ids: torch.Tensor) -> torch.Tensor:
    return ~(image_position_ids == -1).all(dim=-1)


def neighbor_patch_pairs(image_position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coords = image_position_ids[0].detach().cpu()
    valid = valid_patch_mask(image_position_ids)[0].detach().cpu()
    coord_to_index = {
        (int(coord[0]), int(coord[1])): index
        for index, coord in enumerate(coords)
        if bool(valid[index])
    }
    left = []
    right = []
    for (x_coord, y_coord), index in coord_to_index.items():
        for neighbor in ((x_coord + 1, y_coord), (x_coord, y_coord + 1)):
            neighbor_index = coord_to_index.get(neighbor)
            if neighbor_index is not None:
                left.append(index)
                right.append(neighbor_index)
    if not left:
        device = image_position_ids.device
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
    return (
        torch.tensor(left, dtype=torch.long, device=image_position_ids.device),
        torch.tensor(right, dtype=torch.long, device=image_position_ids.device),
    )


def patch_total_variation_loss(pixel_values: torch.Tensor, pair_left: torch.Tensor, pair_right: torch.Tensor) -> torch.Tensor:
    if pair_left.numel() == 0:
        return pixel_values.new_zeros(())
    return F.mse_loss(pixel_values[:, pair_left, :], pixel_values[:, pair_right, :])


def enforce_patch_constraints(pixel_values: torch.Tensor, mask: torch.Tensor) -> None:
    with torch.no_grad():
        pixel_values.clamp_(0.0, 1.0)
        pixel_values.masked_fill_(~mask.unsqueeze(-1), 0.0)


def patches_to_image(
    pixel_values: torch.Tensor,
    image_position_ids: torch.Tensor,
    patch_size: int = 16,
) -> Image.Image:
    values = pixel_values.detach().float().cpu().clamp(0.0, 1.0)[0]
    coords = image_position_ids.detach().cpu()[0]
    valid = valid_patch_mask(image_position_ids.detach().cpu())[0]
    real_coords = coords[valid]
    if real_coords.numel() == 0:
        return Image.new("RGB", (patch_size, patch_size), "black")

    width = int(real_coords[:, 0].max().item()) + 1
    height = int(real_coords[:, 1].max().item()) + 1
    canvas = torch.zeros(height * patch_size, width * patch_size, 3)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    for patch_index in valid_indices:
        x_coord = int(coords[patch_index, 0].item())
        y_coord = int(coords[patch_index, 1].item())
        patch = values[patch_index].reshape(patch_size, patch_size, 3)
        y0 = y_coord * patch_size
        x0 = x_coord * patch_size
        canvas[y0 : y0 + patch_size, x0 : x0 + patch_size, :] = patch

    array = (canvas.numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def save_preview(pixel_values: torch.Tensor, image_position_ids: torch.Tensor, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    patches_to_image(pixel_values, image_position_ids).save(path)
    return str(path)


def save_loss_history(history: list[dict], run_dir: Path) -> tuple[str, str]:
    json_path = run_dir / "loss_history.json"
    csv_path = run_dir / "loss_history.csv"
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history:
        fieldnames = sorted({key for row in history for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
    else:
        csv_path.write_text("", encoding="utf-8")
    return str(json_path), str(csv_path)


def tensor_batch_to_cpu(batch: dict) -> dict:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}


def save_artifact(artifact: dict, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    return str(path)


def load_artifact(path: str | Path) -> dict:
    return torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)


def find_artifact(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    artifact_path = run_dir / "visual_prompt.pt"
    if artifact_path.exists():
        return artifact_path
    matches = sorted(run_dir.glob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"No .pt visual prompt artifact found in {run_dir}")
    return matches[0]


def serializable_config(config: dict) -> dict:
    result = {}
    for key, value in config.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result
