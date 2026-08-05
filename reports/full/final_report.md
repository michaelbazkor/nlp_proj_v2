# Full-cohort run — `full`

Config: `configs/real_full.yaml`  |  seed: 42

## 1. Scale

| Quantity | Value |
|---|---|
| Users (labeled cohort) | 1003 |
| High-risk positives (`suicide >= 3`) | 132 (13.2%) |
| Posts used | 401,360 |
| Unique image keys referenced | 301,324 |
| Images captioned | 165,059 |
| Mean posts/user | 322.9 |
| Mean captioned images/user | 166.7 |
| Mean corpus chars/user | 127,596 |
| Total corpus chars | 128.0 M |

## 2. Image captions (Qwen2.5-VL-7B)

165,059 captions, mean 569 chars (p10 468 / p90 649).

- `10155444438051752_10154842118256752 -id`: The image features a whimsical depiction of the Cheshire Cat from "Alice in Wonderland," characterized by its vibrant purple and white fur, large yellow eyes with vertical pupils, and a wide, toothy grin. The cat is perched on a curved branch, set against a blurred background that suggests a forest or garden setting. The overall mood is playful and slightly surreal, enhanced by the bright, almost neon colors used in 

- `10216033377608183_10215942169008025 -id`: The image features a black background with white text centered in two distinct sections. The top section reads "MY OWN ATTITUDE STRESSES ME OUT." Below it, another line states "I BE MAD AT ME FOR BEING MAD." At the bottom, there's a smaller text "@REBEL CIRCUS" in a lighter font. The overall mood appears reflective and somewhat somber, as the stark contrast between the black background and white text draws attention 

- `10157408940363332_10157250007513332 -id`: The image features a man holding a baby outdoors, likely at a family theme park given the logo in the bottom left corner that reads "Edaville Family Theme Park." The man is wearing a black t-shirt, sunglasses, and a cap, while the baby is dressed in a blue swimsuit with a matching hat. The background includes some greenery and part of a structure, possibly a ride or attraction. The left side of the image has a light 


## 3. Assembled corpus (median-length user, truncated)

`n_posts=139 n_images=91 n_chars=72582`

```
[post] 100%. Imagine that. [post] Artist at work.  On Boxing Day, we paint. 🎨 [image] The image depicts a person engaged in painting at an easel indoors. The individual is wearing a dark t-shirt and appears focused on their work, holding a paintbrush with one hand while the other arm is bent at the elbow. The room has a cozy ambiance with natural light filtering through blinds, creating a soft and inviting atmosphere. Various art supplies, including tubes of paint and a palette, are scattered on the table around the easel. In the background, there's a wooden dresser adorned with decorative items such as a small Christmas tree and a sign that reads "CRAB." The overall mood of the image is calm and creative, suggesting [post] Sure do love where I live. [image] The image depicts a serene natural setting with two Canada geese swimming in a calm body of water. The foreground features wooden posts and a partially submerged structure, possibly part of a dock or pier, suggesting a rustic or rural environment. The background is lush with greenery, including trees and bushes, indicating a peaceful, secluded area. The mood of the image is tranquil and isolated, evoking a sense of calmness and
```

## 4. LLM representations and zero-shot verdicts

| Model | users | blocks tapped | hidden dim | mean chunks/user | `RISK=` parsed | predicted-positive rate | zero-shot AUC | zero-shot F1 |
|---|---|---|---|---|---|---|---|---|
| `qwen3_32b` | 1003 | [20, 40, 60, 64] of 64 | [5120] | 1.89 | 80% | 0.025 | 0.535 | 0.143 |
| `deepseek_r1_distill_qwen_32b` | 1003 | [20, 40, 60, 64] of 64 | [5120] | 1.89 | 79% | 0.010 | 0.518 | 0.078 |
| `gemma4_26b_moe` | 1003 | [20, 30] of 30 | [2816] | 1.89 | 100% | 0.032 | 0.551 | 0.195 |
| `llama_3_3_70b` | 1003 | [20, 40, 60, 80] of 80 | [8192] | 1.89 | 83% | 0.119 | 0.576 | 0.257 |

Zero-shot columns score the model's own `RISK=` label (a single binary decision, so AUC is the balanced-accuracy equivalent) against `y_high`. These are not the paper's classifier; they contextualize the hidden states.

### `qwen3_32b` — sample rationales

- **user** `1008121212695792 -id` label=`1` model_pred=`0`
  > The posts include a traumatic announcement about the death of the user's oldest son, Alvin Figueroa, which indicates profound grief and loss. However, there are no explicit or implicit references to suicidal ideation, self-harm, or severe hopelessness beyond the normal expression of mourning. The majority of the content reflects everyday life, family updates, and community engagement, with no escalation in language suggesting evolving suicidal risk.

- **user** `10100118245537602 -id` label=`0` model_pred=`0`
  > The provided posts primarily consist of images documenting everyday activities involving a child, such as playtime, meals, and outings, with minimal textual content. There are no explicit or implicit references to death, self-harm, emotional distress, or hopelessness. The language and imagery reflect routine, positive, or neutral experiences without evidence of escalating mental health concerns or suicidal ideation.

- **user** `10100123549150429 -id` label=`0` model_pred=`0`
  > The posts primarily reflect daily life, family celebrations, and grief related to a stillborn child (Jailah Elaine), with hashtags like #stillbornstillloved and references to memorializing the child. While there are expressions of emotional pain (e.g., "Where was God during all this?"), these are contextualized within mourning and do not indicate escalating suicidal ideation. No explicit references to self-harm, death, or severe hopelessness beyond grief are present.

- **user** `10100197328125888 -id` label=`0` model_pred=`0`
  > The user's posts primarily focus on entertainment reviews, personal anecdotes, and social events. There are no explicit or implicit references to suicide, self-harm, or severe emotional distress. While one post mentions emotional numbness following a family member's death, it is contextualized as a normal grief response without indicators of suicidal ideation.


### `deepseek_r1_distill_qwen_32b` — sample rationales

- **user** `1008121212695792 -id` label=`1` model_pred=`None`
  > Okay, so I'm trying to assess whether the given Facebook posts indicate an evolving trajectory of suicidal ideation. First, I need to understand what suicidal ideation looks like. From what I know, it often involves feelings of intense emotional pain, hopelessness, social isolation, or feeling like a burden. People might talk about death or self-harm, or show signs of escalating mental health issues. Looking at the posts, the user shares a mix of personal updates, images, and interactions. Let's go through them one by one. The first post mentions being a stay-at-home dad and feeling bored, but

- **user** `10100118245537602 -id` label=`0` model_pred=`0`
  > The user's posts primarily consist of positive and neutral content, focusing on family activities and everyday moments without any explicit or implicit references to suicidal ideation, self-harm, or mental health distress. The language and imagery suggest a normal, engaged family life with no signs of social isolation, hopelessness, or emotional pain.

- **user** `10100123549150429 -id` label=`0` model_pred=`0`
  > The user's posts predominantly reflect positive emotions, strong social connections, and a focus on family and gratitude. While there are mentions of grief related to miscarriages, the user appears to be coping healthily. There are no explicit or implicit references to suicidal ideation, self-harm, or hopelessness.

- **user** `10100197328125888 -id` label=`0` model_pred=`None`
  > The user's posts primarily focus on pop culture, reviews, and social activities, with no


### `gemma4_26b_moe` — sample rationales

- **user** `1008121212695792 -id` label=`1` model_pred=`0`
  > The user's posts primarily document life events, including the tragic death of a child and medical concerns regarding another child, which are significant stressors. However, the language remains focused on grief, parenting, and daily activities rather than expressing personal hopelessness, self-harm, or a desire to die.

- **user** `10100118245537602 -id` label=`0` model_pred=`0`
  > The posts consist almost entirely of mundane, domestic, and celebratory content centered around parenting, family outings, and daily activities. While there are occasional hyperbolic expressions (e.g., "Everyone. Dying." regarding the flu), there are no indicators of hopelessness, social isolation, or suicidal ideation.

- **user** `10100123549150429 -id` label=`0` model_pred=`0`
  > The user's posts primarily focus on family, marriage, celebrations, and navigating significant life events like the loss of a child (stillbirth) and hurricanes. While there is expressed grief regarding "Jailah Elaine" and a brother, the language remains centered on remembrance, love, and resilience rather than hopelessness, entrapment, or suicidal intent.

- **user** `10100197328125888 -id` label=`0` model_pred=`0`
  > The user's posts are overwhelmingly focused on media criticism (movies, TV shows, podcasts) and personal anecdotes regarding local community events and hobbies. While there is a mention of feeling "emotionally numb" following a grandmother's passing, this is presented within the context of grief and is not indicative of an evolving trajectory of suicidal ideation or hopelessness.


### `llama_3_3_70b` — sample rationales

- **user** `1008121212695792 -id` label=`1` model_pred=`0`
  > The user's posts exhibit a mix of emotions and topics, but there are specific instances that raise concern about suicidal ideation. For example, the post about witnessing the death of a child and the subsequent funeral arrangements indicates a period of intense grief and emotional pain. Additionally, the mention of feeling annoyed and angry, as well as the reference to selling an Xbox and games, which could imply a sense of desperation or hopelessness about the future. However, these signals are not consistently present throughout the posts, and there are also expressions of joy, hope, and eng

- **user** `10100118245537602 -id` label=`0` model_pred=`0`
  > The posts do not indicate an evolving trajectory of suicidal ideation. The majority of the content appears to be casual, everyday moments from the user's life, including images of the child playing, eating, and interacting with others. There are no explicit or implicit references to death, self-harm, or severe mental health distress that would suggest suicidal ideation. The tone of the posts is generally light-hearted and playful, reflecting a typical childhood experience.

- **user** `10100123549150429 -id` label=`0` model_pred=`0`
  > The user's posts do not explicitly express suicidal ideation, however, some posts contain language that could be interpreted as emotional pain or distress, such as "Fucking hate this month" or "Bruh this can't be life". Additionally, there are posts that mention loss, grief, or remembrance of a deceased loved one, which may indicate a sense of hopelessness or despair. Nevertheless, the majority of the posts appear to be casual, celebratory, or focused on everyday life, suggesting that the user does not have suicidal ideation.

- **user** `10100197328125888 -id` label=`0` model_pred=`None`
  > Based on the provided Facebook posts, there is no indication of suicidal ideation or evolving trajectory of suicidal thoughts. The user's posts are mostly about movies, TV shows, and other pop culture topics, no explicit or implicit references to death, self-harm, or mental health distress. The tone is generally lighthearted and humorous. Therefore, the risk of suicidal ideation is low.


## 5. STM / MTM cross-validated results

Fusion: 56 blocks from 4 LLMs -> 1024-d. Grid: 504 points x 5 folds x 2 variants.

| Variant | AUC-ROC (mean ± sd [95% CI]) | PR-AUC | F1 | Cohen's d |
|---|---|---|---|---|
| **stm_high** | 0.643 ± 0.030 [0.617, 0.670] | 0.242 ± 0.030 [0.216, 0.268] | 0.228 ± 0.111 [0.131, 0.326] | 0.521 ± 0.114 [0.420, 0.621] |
| **mtm_high** | 0.674 ± 0.058 [0.623, 0.725] | 0.306 ± 0.105 [0.214, 0.397] | 0.148 ± 0.151 [0.016, 0.281] | 0.644 ± 0.232 [0.440, 0.847] |

### `stm_high` per fold

| fold | test AUC | PR-AUC | F1 | d | dev AUC | best hyperparams |
|---|---|---|---|---|---|---|
| 0 | 0.596 | 0.193 | 0.122 | 0.342 | 0.843 | L=2 n=256 tanh lr=0.005 ep=2500 |
| 1 | 0.642 | 0.269 | 0.150 | 0.515 | 0.888 | L=1 n=128 tanh lr=0.001 ep=2500 |
| 2 | 0.651 | 0.241 | 0.171 | 0.547 | 0.862 | L=1 n=16 tanh lr=0.05 ep=2500 |
| 3 | 0.648 | 0.264 | 0.357 | 0.538 | 0.819 | L=3 n=512 tanh lr=0.001 ep=1000 |
| 4 | 0.680 | 0.243 | 0.340 | 0.660 | 0.817 | L=2 n=32 tanh lr=0.001 ep=5000 |

### `mtm_high` per fold

| fold | test AUC | PR-AUC | F1 | d | dev AUC | best hyperparams |
|---|---|---|---|---|---|---|
| 0 | 0.616 | 0.248 | 0.262 | 0.417 | 0.873 | L=2 n=128 tanh lr=0.01 ep=2500 |
| 1 | 0.620 | 0.242 | 0.333 | 0.431 | 0.852 | L=2 n=256 tanh lr=0.001 ep=5000 |
| 2 | 0.751 | 0.471 | 0.000 | 0.960 | 0.899 | L=3 n=32 sigmoid lr=0.001 ep=2500 |
| 3 | 0.710 | 0.348 | 0.000 | 0.782 | 0.819 | L=3 n=64 sigmoid lr=0.001 ep=1000 |
| 4 | 0.671 | 0.219 | 0.146 | 0.627 | 0.828 | L=2 n=64 tanh lr=0.001 ep=2500 |

## 6. Findings and limitations

- **Multi-task helps.** MTM reaches AUC 0.674 vs STM 0.643; the auxiliary personality / psychosocial / psychiatric heads add signal, which is the direction Ophir et al. report.

- **Model selection overfits the dev split.** Mean dev AUC is 0.846 (STM) and 0.854 (MTM) against test 0.643 / 0.674. With 504 grid points scored on 142 dev users (~19 positives), the winning configuration is partly selected on dev noise. The test numbers are the honest ones; a narrower grid or repeated inner CV would shrink the gap.

- **F1 is unstable and near-uninformative at this base rate.** 13.2% positives with a 0.5 threshold puts some folds at F1 = 0 while their AUC is 0.75; AUC / PR-AUC / Cohen's d are the metrics to read.

- **Thinking models get truncated.** Qwen3 and R1-Distill spend their 512-token budget inside `<think>`, so a `RISK=` verdict was recoverable for only 79–80% of users (Gemma: 100%). For those users the `final_pred` tap sits mid-reasoning rather than after a decision, which weakens that one block; `cot` and the prompt-side blocks are unaffected.

- **Image coverage is bounded by the archive, not the pipeline.** Posts reference 301,324 images but `pics.zip` holds 165,664 files, so 165,059 (55%) could be captioned. The rest are referenced-but-absent blobs.

- **Zero-shot prompting alone is weak.** Every model's own `RISK=` verdict lands near chance (AUC 0.52–0.58) and is heavily conservative, while the same models' hidden states support AUC ~0.67. The representation, not the verbalized answer, carries the signal — the premise of the approach.

- Earlier `artifacts/baseline_original/*_metrics.json` come from 30-user POC runs with small models and are too noisy to compare against these results.

## 7. Measured run cost

**Captioning** (4 GPU shards): 164,967 images in 2.22 h wall, 20.9 img/s aggregate (5.22 img/s per GPU). The unbatched captioner measured 0.86 img/s on one GPU, so this is ~24x faster end to end.

**Representation extraction** (one model per GPU; the slower models were re-sharded onto freed GPUs as faster ones finished):

| Model | workers | s/user | peak GPU |
|---|---|---|---|
| `deepseek_r1_distill_qwen_32b` | 3 | 53.0 | 74.7 GiB |
| `gemma4_26b_moe` | 1 | 15.5 | 65.3 GiB |
| `llama_3_3_70b` | 3 | 48.4 | 87.2 GiB |
| `qwen3_32b` | 3 | 52.3 | 74.4 GiB |

**Training**: 10 (variant, fold) grid searches of 504 configs each, 40 min total (4.0 min each) on 48 workers with wide nets routed to GPUs and narrow nets to single CPU threads.

## 8. Reproduce

```bash
# captions (4 GPUs, batched)
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/caption_shard.py \
    --config configs/real_full.yaml --shard $i --num-shards 4 --batch-size 48 & done; wait
python scripts/merge_caption_shards.py --config configs/real_full.yaml
PYTHONPATH=src python -m ssr.cli --config configs/real_full.yaml corpus
# represent: one model per GPU
PYTHONPATH=src python -m ssr.cli --config configs/real_full.yaml represent   # SSR_ONLY_MODELS=<name>
python scripts/train_parallel.py --config configs/real_full.yaml
python scripts/make_report.py --config configs/real_full.yaml
```
