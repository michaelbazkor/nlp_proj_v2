# Stage examples (POC run)

## 0. Cohort row

```
UserId          204720893660851 -id
suicide                         2.0
y_general                         1
y_high                            0
status_posts                     34
age                            27.0
gender_label                 Female
PHQ9                           11.0
GAD                            17.0
BFI_N                           7.0
Lonely                         30.0
```

## 1. Raw cleaned posts (sample)

- **PostType=photo** has_image=True
  text: `Once I pop that pill deuces ✌😭😭😂😂`
  blob_guid: `d30bc494-8c65-e811-80c3-000d3a117954`
- **PostType=photo** has_image=True
  text: `Bye 🏃‍♀️🏃‍♀️`
  blob_guid: `655cbc9a-8c65-e811-80c3-000d3a117954`
- **PostType=video** has_image=True
  text: `🤣🤣🤣🤣🤣`
  blob_guid: `635dbc9a-8c65-e811-80c3-000d3a117954`

## 2. Assembled corpus (truncated)

n_posts=30 n_images=8 n_chars=2539

```
[post] It was like I never left 😡😡 waste of time [post] I do better alone ✌ [post] 🤣🤣🤣🤣 [post] My patient finally came home today 😊 Still tired tho 😐 [post] Alright I work this weekend I dont wanna see nobody I know in the ER this weekend...stay home and take Tylenol 😂😂😂 [post] It feels good knowing you're  appreciated 😍😍 [post] IMAGE_DESCRIPTIONS: The image shows a social-media image of a young woman, presumably a singer or a celebrity, who is captured in a moment of intense social media interaction. She is shown in a close-up shot, with her hair styled in loose waves, and her expression is one of concentration and determination. Her expression is [post] 😎😎 [post] Pepper trying to move the pillow to cover up his mess lol, he just put the sock there when i walked in [post] IMAGE_DESCRIPTIONS: Imma start telling dudes I have 4 kids so they'll leave me alone. [post] 🤣🤣🤣🤣🤣 [post] Say what now!! 🤔🤔 [post] IMAGE_DESCRIPTIONS: A social media image of a person with a lot of text on it. [post] Once I pop that pill deuces ✌😭😭😂😂 IMAGE_DESCRIPTIONS: This social-media image depicts a group of four people seated at a table in a kitchen. The individuals are engaged in a conversation, with one person looking at the camera and the others looking at the camera. The table is set with a variety of food items, including a bowl of soup, a plate of vegetables, [post] Yes I did and he loves them lol 😂😂😂 Yall know these my churn 😍😍 [post] All ima say is stay off them drugs 👀👀 [post] Bye 🏃‍♀️🏃‍♀️ IMA
```

## 3. Raw representation blocks

### Model `qwen3_0_6b`

- `1:input_only`: shape=(1024,) mean=0.0005 std=0.1151
- `1:last_prompt`: shape=(1024,) mean=0.0041 std=0.2191
- `1:cot`: shape=(1024,) mean=-0.0004 std=0.1262
- `1:final_pred`: shape=(1024,) mean=0.0028 std=0.2469
- `10:input_only`: shape=(1024,) mean=0.0148 std=0.6176
- `10:last_prompt`: shape=(1024,) mean=0.0215 std=0.7559
- `10:cot`: shape=(1024,) mean=0.0121 std=0.6443
- `10:final_pred`: shape=(1024,) mean=0.0064 std=0.9117

### Model `smollm2_360m`

- `1:input_only`: shape=(960,) mean=-0.0536 std=1.3345
- `1:last_prompt`: shape=(960,) mean=-0.0745 std=1.5683
- `1:cot`: shape=(960,) mean=-0.1045 std=1.6994
- `1:final_pred`: shape=(960,) mean=-0.0545 std=1.8670
- `11:input_only`: shape=(960,) mean=-0.1311 std=4.2219
- `11:last_prompt`: shape=(960,) mean=-0.2517 std=4.2095
- `11:cot`: shape=(960,) mean=-0.1761 std=4.4514
- `11:final_pred`: shape=(960,) mean=-0.1525 std=5.2557

## 4. Fused 1024-d vector (fit on all POC users for illustration)

k_per_block=29 n_blocks=32 out_dim=1024

vector[:16] = `[-5.361884117126465, -10.766600608825684, 8.963253021240234, 11.853469848632812, -14.224342346191406, -8.543961524963379, -1.324390172958374, 0.6574892401695251, -1.756801724433899, 7.003340721130371, 3.373141288757324, -4.590076923370361, 6.567163944244385, -8.987096786499023, -7.079833984375, 1.4560657739639282]`

vector mean=0.0650 std=5.9350 l2=189.9316

## 5. CV metrics summary

| Variant | AUC-ROC | PR-AUC | F1 | Cohen's d |
|---------|---------|--------|----|-----------|
| stm_general | 0.397 | 0.474 | 0.200 | -0.480 |
| stm_high | 0.412 | 0.438 | 0.200 | -0.328 |
| mtm_general | 0.629 | 0.573 | 0.167 | 0.529 |
| mtm_high | 0.287 | 0.256 | 0.080 | -2.015 |

## 6. Experiment comparison (30-user POC; noisy by design)

| Variant | text_only AUC | standard AUC |
|---------|---------------|--------------|
| stm_general | 0.506 | 0.397 |
| stm_high | 0.550 | 0.412 |
| mtm_general | 0.450 | 0.629 |
| mtm_high | 0.738 | 0.287 |

Notes:
- Labels: high risk only — `y_high=(suicide>=3)`. Image captions use per-image `[image]` markers.
- POC models: Qwen3-0.6B + SmolLM2-360M; captioner: SmolVLM-256M (158 captions).
- Fusion: per-block PCA to 1024 (fold-safe). Architecture matches H100 `configs/real.yaml`.
- Metrics with N=30 are unstable; purpose is pipeline proof, not paper-comparable AUCs.
