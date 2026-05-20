# How VisualPromptCodec Works

## Gemma Architecture Map

This is the part of Gemma that matters for this implementation, based on the model printout in `scripts/test_model_inference.py`.

```text
Gemma4ForConditionalGeneration
|
+-- model: Gemma4Model
|   |
|   +-- vision_tower: Gemma4VisionModel
|   |   |
|   |   +-- input from processor
|   |   |      pixel_values: [1, 2520, 768]
|   |   |      image_position_ids: [1, 2520, 2]
|   |   |
|   |   +-- patch_embedder: Gemma4VisionPatchEmbedder
|   |   |      input_proj: Linear(768 -> 768)
|   |   |      important here:
|   |   |        768 = 16 * 16 * 3 flattened RGB patch
|   |   |        this is the tensor we optimize in patch space
|   |   |
|   |   +-- encoder: Gemma4VisionEncoder
|   |   |      layers: 16 x Gemma4VisionEncoderLayer
|   |   |      hidden width: 768
|   |   |      attention projections: 768 -> 768
|   |   |      MLP: 768 -> 3072 -> 768
|   |   |
|   |   +-- pooler: Gemma4VisionPooler
|   |          compresses valid patch features into image soft tokens
|   |
|   +-- embed_vision: Gemma4MultimodalEmbedder
|   |      embedding_pre_projection_norm: RMSNorm(768)
|   |      embedding_projection: Linear(768 -> 1536)
|   |      important here:
|   |        this is the bridge from vision space into language-model space
|   |
|   +-- language_model: Gemma4TextModel
|   |   |
|   |   +-- embed_tokens
|   |   |      vocabulary size: 262144
|   |   |      text hidden width: 1536
|   |   |
|   |   +-- layers: 35 x Gemma4TextDecoderLayer
|   |   |      hidden width: 1536
|   |   |      selected alignment taps for this project:
|   |   |        layer 16 -> hidden vector [1, 1536]
|   |   |        layer 20 -> hidden vector [1, 1536]  primary target
|   |   |        layer 24 -> hidden vector [1, 1536]
|   |   |
|   |   +-- final text hidden state
|   |          shape per token: [1, 1536]
|   |
|   +-- audio_tower / embed_audio
|          present in the model, not used by this implementation
|
+-- lm_head
       Linear(1536 -> 262144)
       converts final language hidden states into token logits
```

The optimizer only changes the visual prompt tensor before the vision tower:

```text
optimized variable:
  pixel_values[valid_patches]  shape approximately [2304, 768]

frozen model path:
  pixel_values
    -> vision_tower
    -> embed_vision
    -> language_model layers 16, 20, 24
    -> hidden-state alignment loss
```

The text prompt branch is used only to produce target hidden vectors:

```text
text prompt
  -> embed_tokens
  -> language_model layers 16, 20, 24
  -> target hidden vectors [1, 1536]
```

The visual prompt branch tries to match those targets:

```text
optimized image patches + "Respond:"
  -> vision_tower
  -> embed_vision
  -> language_model layers 16, 20, 24
  -> visual hidden vectors [1, 1536]
```

## Image Processor Operations

Gemma's image processor turns a normal RGB image into the patch tensor consumed by the vision tower:

```text
RGB/PIL image
  -> convert to RGB tensor
  -> aspect-ratio-preserving resize
  -> rescale pixel values to [0, 1]
  -> split into 16 x 16 RGB patches
  -> flatten each patch to 16 * 16 * 3 = 768 values
  -> pad patch rows to 280 * 3^2 = 2520
  -> pixel_values: [1, 2520, 768]
  -> patch_embedder.input_proj: Linear(768 -> 768)
  -> vision encoder
  -> embed_vision: Linear(768 -> 1536)
  -> language model
```

The resize is driven by Gemma's patch budget, not by a fixed target size. For an input image with height `H` and width `W`:

```text
max_patches = max_soft_tokens * pooling_kernel_size^2
            = 280 * 3^2
            = 2520

target_pixels = max_patches * patch_size^2
              = 2520 * 16^2

scale = sqrt(target_pixels / (H * W))

ideal_height = H * scale
ideal_width  = W * scale
```

Then both dimensions are rounded down to multiples of:

```text
pooling_kernel_size * patch_size = 3 * 16 = 48
```

For the default `448 x 448` seed image:

```text
target size = 768 x 768
patch grid  = 48 x 48
valid patches = 2304
padded rows = 2520 - 2304 = 216
```

There are now two optimizer variants:

```text
Patch-space optimizer:
  starts after image processing
  optimizes pixel_values directly
  optimized variable: [1, 2520, 768]

RGB optimizer:
  starts before image processing
  optimizes a normal RGB tensor
  optimized variable: [1, 3, H, W]
  differentiably converts RGB -> pixel_values each step
  also matches the text prompt's next-token distribution by default
```

By default, the RGB optimizer uses the same target-size math and patch layout as Gemma's processor, but it disables antialiased resize during optimization so gradients work on Apple MPS. Pass `--resize-antialias` to use antialiased resize when running on a backend that supports that backward pass.

## Methodology

1. Pick a text prompt.

   Example:

   ```text
   Who is the president of USA?
   ```

   This is the behavior we want Gemma to reproduce from an image-like input.

2. Run Gemma normally with the text prompt.

   The optimizer sends Gemma the real text prompt through the normal text path:

   ```text
   User: Who is the president of USA?
   ```

   It then extracts Gemma's internal hidden states at selected language-model layers. The current default layers are:

   ```text
   16, 20, 24
   ```

   Gemma's text hidden size is `1536`, so each extracted target vector has shape:

   ```text
   [1, 1536]
   ```

   There is one vector per selected layer. These hidden vectors become the target representation.

3. Create a visual prompt candidate.

   The optimizer creates a seed image, currently a plain gray image, and passes it through Gemma's image processor.

   Gemma does not directly receive normal RGB pixels. The processor converts the image into patch-space tensors:

   ```text
   pixel_values: [1, 2520, 768]
   ```

   These dimensions come from Gemma's image processor and model config:

   ```text
   batch size = 1
   patch size = 16 x 16
   channels = 3
   one patch vector = 16 * 16 * 3 = 768
   max soft image tokens = 280
   pooling kernel = 3
   max patch rows = 280 * 3 * 3 = 2520
   ```

   So the processed image tensor shape is:

   ```text
   [batch, max_patch_rows, patch_vector_width]
   = [1, 2520, 768]
   ```

   In the default `448 x 448` seed image case, the processor resizes the image to a `48 x 48` patch grid:

   ```text
   valid patches = 48 * 48 = 2304
   padded patches = 2520 - 2304 = 216
   ```

   These are flattened image patches. Only valid patches are optimized; padding patches stay zero.

4. Add the tiny visual-side anchor.

   The current visual prompt path is:

   ```text
   <optimized image>
   Respond:
   ```

   Gemma does not see the original prompt text in this branch. It only sees the optimized image plus the tiny text anchor `Respond:`.

5. Freeze Gemma.

   Gemma's weights are not trained. Only the image patch tensor is trainable.

   The optimization question is:

   ```text
   Can we change the image patches so Gemma internally behaves like it saw the text prompt?
   ```

6. Compare hidden states.

   For every optimization step, the script runs Gemma two ways.

   Text path:

   ```text
   Who is the president of USA?
   ```

   Visual path:

   ```text
   <current image patches>
   Respond:
   ```

   It extracts hidden states from layers `16`, `20`, and `24` at the final context token.

   Both branches produce language-model hidden vectors with the same shape:

   ```text
   text_hidden[layer]:   [1, 1536]
   visual_hidden[layer]: [1, 1536]
   ```

   The image starts in vision patch space as `[1, 2520, 768]`, then Gemma's vision tower and multimodal embedder project image information into the language model's `1536`-wide hidden space before the hidden-state comparison is made.

7. Optimize the visual prompt.

   The loss tries to make the visual hidden states match the text hidden states. Let:

   ```text
   L = {16, 20, 24}
   h_text_l = text hidden vector at layer l, shape [1, 1536]
   h_img_l  = visual hidden vector at layer l, shape [1, 1536]
   c_l      = cosine_similarity(h_img_l, h_text_l)
   ```

   The alignment loss is:

   ```text
   alignment_loss =
       sum over l in L:
           w_l * (1 - c_l)
   ```

   The default weights are:

   ```text
   w_16 = 1
   w_20 = 2
   w_24 = 1
   ```

   Layer `20` is weighted most strongly because it is the main semantic target.

   The patch-space optimizer also adds light regularization:

   ```text
   l2_loss = mean((current_valid_patches - seed_valid_patches)^2)

   tv_loss = mean((patch_i - patch_j)^2)
             for neighboring valid patch positions i, j
   ```

   ```text
   patch_total_loss =
       alignment_loss
     + l2_weight * l2_loss
     + tv_weight * tv_loss
   ```

   With default weights:

   ```text
   l2_weight = 0.001
   tv_weight = 0.001
   ```

   These penalties help prevent the patch tensor from becoming completely wild.

   The RGB optimizer uses the same hidden-state alignment, but adds direct behavior losses. First, it matches the next-token distribution from the text prompt:

   ```text
   logit_kl_loss =
       KL(
         softmax(text_next_token_logits / temperature),
         softmax(visual_next_token_logits / temperature)
       ) * temperature^2
   ```

   It also adds a stronger teacher-forced answer loss. The script first asks Gemma to answer the text prompt, then uses those generated answer tokens as the supervised target for the visual branch:

   ```text
   text prompt
     -> Gemma generate
     -> target answer token ids

   optimized RGB image + "Respond:" + target answer prefix
     -> Gemma forward pass
     -> cross entropy on target answer tokens only
   ```

   The RGB total loss is:

   ```text
   rgb_total_loss =
       alignment_loss
     + logit_kl_weight * logit_kl_loss
     + target_ce_weight * target_ce_loss
     + l2_weight * l2_loss
     + tv_weight * tv_loss
   ```

   Default RGB behavior weights:

   ```text
   logit_kl_weight = 0.1
   target_ce_weight = 1.0
   target_max_new_tokens = 64
   ```

   These behavior losses are important because hidden-state matching can look numerically good while the model still behaves like an image-captioning model. Matching next-token logits and teacher-forced answer tokens gives the visual prompt direct pressure toward the text prompt's answer behavior.

8. Save the learned visual prompt.

   At the end, the optimizer saves:

   ```text
   visual_prompt.pt
   visual_prompt_preview.png
   loss_history.json
   loss_history.csv
   config.json
   ```

   The `.pt` file contains the optimized patch tensor, original prompt, visual-side anchor text, layer choices, model path, and optimization history.

   RGB optimizer artifacts also contain:

   ```text
   final_rgb_image
   rgb_preview_path
   target_response_ids
   ```

9. Evaluate four conditions.

   The evaluator compares:

   ```text
   text_prompt
   optimized_visual_prompt
   rendered_text_image
   random_patch_image
   ```

   Meaning:

   - `text_prompt`: Gemma gets the real text prompt.
   - `optimized_visual_prompt`: Gemma gets the learned image patches plus `Respond:`.
   - `rendered_text_image`: Gemma gets an image with the prompt visibly written on it, plus `Respond:`.
   - `random_patch_image`: Gemma gets random image patches plus `Respond:`.

10. Interpret success.

    The visual prompt is working when:

    ```text
    optimized_visual_prompt behaves more like text_prompt
    than random_patch_image does
    ```

    For example, if the text prompt asks for a haiku and the optimized visual prompt produces a Dubai poem, that is a partial success.

    If it follows exact format and intent, that is stronger success.

## Current Limitation

Hidden-state alignment alone is not enough. In early RGB experiments, the optimizer reached reasonable hidden-state cosine scores while Gemma still described the optimized image as a map, puzzle, or texture instead of answering the prompt.

The RGB optimizer now works better when target-response cross entropy is enabled. A tested command:

```bash
PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mainenv/bin/python scripts/optimize_visual_prompt_rgb.py \
  --prompt "Who is the president of the United States?" \
  --steps 100 \
  --lr 0.01 \
  --layers 16,20,24 \
  --logit-kl-weight 0.5 \
  --target-ce-weight 5.0 \
  --target-max-new-tokens 64
```

In that run, the teacher-forced answer loss dropped sharply:

```text
target_ce_loss:
step 1    3.5435
step 50   0.0569
step 100  0.0260
```

The evaluation then matched the text-prompt answer:

```text
text_prompt:
  The current President of the United States is **Joe Biden**.

optimized_visual_prompt:
  The current President of the United States is **Joe Biden**.

random_patch_image:
  image-description / texture response
```

This means the RGB visual prompt learned to reproduce Gemma's local behavior for that prompt. It does not mean the answer is current-world factual truth; it means the visual prompt reproduced the frozen local model's text-prompt behavior.
