#load from ../models/gemma-4-E2B-it
# generate output for a sample input prompt
# print output

from transformers import AutoProcessor, AutoModelForCausalLM

import torch

# Load model and processor
model_path = "../models/gemma-4-E2B-it"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)
print("device : ", model.device)
print("dtype : ", model.dtype)
# Sample input prompt
# prompt = "Write a poem about Dubai."
# # Prompt
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt},
# ]

# # Process input
# text = processor.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True, 
#     enable_thinking=False
# )
# inputs = processor(text=text, return_tensors="pt").to(model.device)
# input_len = inputs["input_ids"].shape[-1]

# # Generate output
# outputs = model.generate(**inputs, max_new_tokens=1024)
# response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# # Parse output
# print(processor.parse_response(response))
'''
## Output:
device :  mps:0
dtype :  torch.bfloat16
{'role': 'assistant', 'content': "## Desert Dreams and Towers Bright\n\nWhere the sands of Arabia softly sigh,\nBeneath a sun that burns the endless sky,\nA city rises, born of vision bold,\nA story in glass, a tale to be told.\n\nFrom desert dust, a spectacle takes hold,\nA tapestry of silver, brave and old.\nThe gleaming towers pierce the azure air,\nA testament to ambition, rich and rare.\n\nThe Burj Khalifa, a needle sharp and high,\nThat scrapes the clouds against the boundless eye.\nA monument to human will and grace,\nReflecting light in that magnificent space.\n\nThen turn to water, crystal, cool, and deep,\nWhere luxury and splendor softly sleep.\nThe yachts glide by with effortless, sleek grace,\nIn waters kissed by sunshine on their face.\n\nThe souks unfold, a labyrinth of scent,\nOf spices rare and treasures heaven-sent.\nA thousand hues of silk and gleaming gold,\nA history in every tale untold.\n\nFrom ancient roots, a future takes its hold,\nA modern marvel, brave and ever bold.\nA fusion born of East and Western might,\nA dazzling beacon burning ever bright.\n\nThe nights ignite with fireworks' vibrant spray,\nPainting the darkness in a dazzling way.\nA symphony of light, a vibrant sound,\nWhere dreams take flight and wonder can be found.\n\nOh, Dubai, city of dazzling sheen,\nWhere dreams are built and majesty is seen.\nA vibrant pulse in a desert land,\nA shining jewel held within God's hand."}
'''

print(model)
'''
Gemma4ForConditionalGeneration(
  (model): Gemma4Model(
    (vision_tower): Gemma4VisionModel(
      (patch_embedder): Gemma4VisionPatchEmbedder(
        (input_proj): Linear(in_features=768, out_features=768, bias=False)
      )
      (encoder): Gemma4VisionEncoder(
        (rotary_emb): Gemma4VisionRotaryEmbedding()
        (layers): ModuleList(
          (0-15): 16 x Gemma4VisionEncoderLayer(
            (self_attn): Gemma4VisionAttention(
              (q_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=768, out_features=768, bias=False)
              )
              (k_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=768, out_features=768, bias=False)
              )
              (v_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=768, out_features=768, bias=False)
              )
              (o_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=768, out_features=768, bias=False)
              )
              (q_norm): Gemma4RMSNorm()
              (k_norm): Gemma4RMSNorm()
              (v_norm): Gemma4RMSNorm()
            )
            (mlp): Gemma4VisionMLP(
              (gate_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=768, out_features=3072, bias=False)
              )
              (up_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=768, out_features=3072, bias=False)
              )
              (down_proj): Gemma4ClippableLinear(
                (linear): Linear(in_features=3072, out_features=768, bias=False)
              )
              (act_fn): GELUTanh()
            )
            (input_layernorm): Gemma4RMSNorm()
            (post_attention_layernorm): Gemma4RMSNorm()
            (pre_feedforward_layernorm): Gemma4RMSNorm()
            (post_feedforward_layernorm): Gemma4RMSNorm()
          )
        )
      )
      (pooler): Gemma4VisionPooler()
    )
    (language_model): Gemma4TextModel(
      (embed_tokens): Gemma4TextScaledWordEmbedding(262144, 1536, padding_idx=0)
      (layers): ModuleList(
        (0-3): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (k_norm): Gemma4RMSNorm()
            (v_norm): Gemma4RMSNorm()
            (k_proj): Linear(in_features=1536, out_features=256, bias=False)
            (v_proj): Linear(in_features=1536, out_features=256, bias=False)
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (up_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (4): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (k_norm): Gemma4RMSNorm()
            (v_norm): Gemma4RMSNorm()
            (k_proj): Linear(in_features=1536, out_features=512, bias=False)
            (v_proj): Linear(in_features=1536, out_features=512, bias=False)
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (up_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (5-8): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (k_norm): Gemma4RMSNorm()
            (v_norm): Gemma4RMSNorm()
            (k_proj): Linear(in_features=1536, out_features=256, bias=False)
            (v_proj): Linear(in_features=1536, out_features=256, bias=False)
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (up_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (9): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (k_norm): Gemma4RMSNorm()
            (v_norm): Gemma4RMSNorm()
            (k_proj): Linear(in_features=1536, out_features=512, bias=False)
            (v_proj): Linear(in_features=1536, out_features=512, bias=False)
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (up_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (10-13): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (k_norm): Gemma4RMSNorm()
            (v_norm): Gemma4RMSNorm()
            (k_proj): Linear(in_features=1536, out_features=256, bias=False)
            (v_proj): Linear(in_features=1536, out_features=256, bias=False)
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (up_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (14): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (k_norm): Gemma4RMSNorm()
            (v_norm): Gemma4RMSNorm()
            (k_proj): Linear(in_features=1536, out_features=512, bias=False)
            (v_proj): Linear(in_features=1536, out_features=512, bias=False)
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (up_proj): Linear(in_features=1536, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (15-18): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (19): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (20-23): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (24): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (25-28): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (29): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (30-33): 4 x Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=2048, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=2048, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
        (34): Gemma4TextDecoderLayer(
          (self_attn): Gemma4TextAttention(
            (q_proj): Linear(in_features=1536, out_features=4096, bias=False)
            (q_norm): Gemma4RMSNorm()
            (o_proj): Linear(in_features=4096, out_features=1536, bias=False)
          )
          (mlp): Gemma4TextMLP(
            (gate_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (up_proj): Linear(in_features=1536, out_features=12288, bias=False)
            (down_proj): Linear(in_features=12288, out_features=1536, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma4RMSNorm()
          (post_attention_layernorm): Gemma4RMSNorm()
          (pre_feedforward_layernorm): Gemma4RMSNorm()
          (post_feedforward_layernorm): Gemma4RMSNorm()
          (act_fn): GELUTanh()
          (per_layer_input_gate): Linear(in_features=1536, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=1536, bias=False)
          (post_per_layer_input_norm): Gemma4RMSNorm()
        )
      )
      (norm): Gemma4RMSNorm()
      (rotary_emb): Gemma4TextRotaryEmbedding()
      (embed_tokens_per_layer): Gemma4TextScaledWordEmbedding(262144, 8960, padding_idx=0)
      (per_layer_model_projection): Linear(in_features=1536, out_features=8960, bias=False)
      (per_layer_projection_norm): Gemma4RMSNorm()
    )
    (audio_tower): Gemma4AudioModel(
      (subsample_conv_projection): Gemma4AudioSubSampleConvProjection(
        (layer0): Gemma4AudioSubSampleConvProjectionLayer(
          (conv): Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True, bias=False)
          (act): ReLU()
        )
        (layer1): Gemma4AudioSubSampleConvProjectionLayer(
          (conv): Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True, bias=False)
          (act): ReLU()
        )
        (input_proj_linear): Linear(in_features=1024, out_features=1024, bias=False)
      )
      (rel_pos_enc): Gemma4AudioRelPositionalEncoding()
      (layers): ModuleList(
        (0-11): 12 x Gemma4AudioLayer(
          (feed_forward1): Gemma4AudioFeedForward(
            (ffw_layer_1): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=4096, bias=False)
            )
            (ffw_layer_2): Gemma4ClippableLinear(
              (linear): Linear(in_features=4096, out_features=1024, bias=False)
            )
            (pre_layer_norm): Gemma4RMSNorm()
            (post_layer_norm): Gemma4RMSNorm()
            (act_fn): SiLUActivation()
          )
          (feed_forward2): Gemma4AudioFeedForward(
            (ffw_layer_1): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=4096, bias=False)
            )
            (ffw_layer_2): Gemma4ClippableLinear(
              (linear): Linear(in_features=4096, out_features=1024, bias=False)
            )
            (pre_layer_norm): Gemma4RMSNorm()
            (post_layer_norm): Gemma4RMSNorm()
            (act_fn): SiLUActivation()
          )
          (self_attn): Gemma4AudioAttention(
            (q_proj): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=1024, bias=False)
            )
            (k_proj): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=1024, bias=False)
            )
            (v_proj): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=1024, bias=False)
            )
            (post): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=1024, bias=False)
            )
            (relative_k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          )
          (lconv1d): Gemma4AudioLightConv1d(
            (linear_start): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=2048, bias=False)
            )
            (linear_end): Gemma4ClippableLinear(
              (linear): Linear(in_features=1024, out_features=1024, bias=False)
            )
            (depthwise_conv1d): Gemma4AudioCausalConv1d(1024, 1024, kernel_size=(5,), stride=(1,), groups=1024, bias=False)
            (pre_layer_norm): Gemma4RMSNorm()
            (conv_norm): Gemma4RMSNorm()
            (act_fn): SiLUActivation()
          )
          (norm_pre_attn): Gemma4RMSNorm()
          (norm_post_attn): Gemma4RMSNorm()
          (norm_out): Gemma4RMSNorm()
        )
      )
      (output_proj): Linear(in_features=1024, out_features=1536, bias=True)
    )
    (embed_vision): Gemma4MultimodalEmbedder(
      (embedding_projection): Linear(in_features=768, out_features=1536, bias=False)
      (embedding_pre_projection_norm): Gemma4RMSNorm()
    )
    (embed_audio): Gemma4MultimodalEmbedder(
      (embedding_projection): Linear(in_features=1536, out_features=1536, bias=False)
      (embedding_pre_projection_norm): Gemma4RMSNorm()
    )
  )
  (lm_head): Linear(in_features=1536, out_features=262144, bias=False)
)
'''