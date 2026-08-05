# llama_n5 smoke batch report (no NN train)

## GPU connectivity

- 4× NVIDIA RTX PRO 6000 Blackwell (~98 GiB each), driver 595.58.03, CUDA 13.2 / torch cu128
- `torch.cuda.is_available()=True`; smoke matmul on cuda:0 succeeded
- Idle after caption stage (0 MiB used)

## Pipeline status

| Stage | Status | Notes |
|---|---|---|
| cohort | OK | n=5 stratified (`seed=42`), 1/5 high-risk (`y_high`) |
| posts | OK | capped `max_posts_per_user=40` → 200 posts |
| captions (image dubbing) | OK | Qwen2.5-VL-7B; 41 captions (19 missing from zip) |
| corpus | OK | mean ~9.9k chars/user; `[image]` captions embedded |
| represent (Llama-3.3-70B) | BLOCKED | gated HF repo → **401 Unauthorized** (no `HF_TOKEN`) |
| train STM/MTM | SKIPPED | per request |

Config: `configs/real_llama_n5.yaml` → artifacts under `artifacts/llama_n5/`.

Smoke caps (to keep a 5-user batch tractable): `max_posts_per_user=40`, `max_images_per_user=12`.

## Cohort (5 stratified users)

```
               UserId  suicide  y_high  status_posts  age gender_label  PHQ9  GAD  Lonely
10100271303188713 -id      0.0       0            12 30.0       Female   9.0  9.0    15.0
10212467798318978 -id      3.0       1           137 28.0       Female  12.0 25.0    30.0
10156459020503223 -id      1.0       0            65 29.0         Male   4.0  8.0    17.0
10213660402236997 -id      0.0       0            31 30.0       Female   8.0 11.0    22.0
10155627703976545 -id      0.0       0           104 36.0       Female   0.0 12.0    22.0
```

## Corpus sizes

```
               UserId  n_posts  n_images  n_chars
10100271303188713 -id       40        11    16561
10155627703976545 -id       38         7     7171
10156459020503223 -id       39         4    10707
10212467798318978 -id       32         7     7243
10213660402236997 -id       15        12     7662
```

## Image dubbing examples (Qwen2.5-VL captions)

Caption meta: `model=Qwen/Qwen2.5-VL-7B-Instruct`, `n_new=41`, `n_requested=60`, `n_missing_in_zip=19`.

### User `10100271303188713 -id` — y_high=0, suicide=0.0
- **PostType=photo** text=`Ryker is doing much better today!`
  - caption: The image features a young child wearing a red helmet with black spiderweb patterns, suggesting they might be engaged in an activity like skateboarding or biking. The child is indoors, with a neutral-colored wall and a piece of furniture visible in the background...
- **PostType=photo** text=`(empty)`
  - caption: The image depicts a close-up of two individuals, likely a couple, sharing a tender moment. The person on the left, wearing a white outfit with a sheer veil adorned with a flower, appears to be smiling warmly...

### User `10155627703976545 -id` — y_high=0, suicide=0.0
- **PostType=photo** text=`🌞🌞🌸`
  - caption: The image shows a person sitting inside a car, with the window partially open, allowing some light to enter. The individual has long, straight blonde hair and is wearing a light-colored t-shirt...
- **PostType=photo** text=`What we do while we wait for the call from the vet to pick up sweet Raymond 💜`
  - caption: The image shows a cozy indoor scene where a person is lying down, partially covered by a brown and black patterned blanket... A dog... is curled up next to the person...

### User `10156459020503223 -id` — y_high=0, suicide=1.0
- **PostType=photo** text=`¡Carnaval!`
  - caption: The image depicts a person wearing a wide-brimmed hat adorned with colorful tassels... body painted black... festive attire...
- **PostType=photo** text=`Even the major city streets in Medellín are beautiful...`
  - caption: The image depicts a busy urban street scene with multiple lanes of traffic... Trees line both sides of the road...

### User `10212467798318978 -id` — y_high=1, suicide=3.0
- **PostType=photo** text=`(empty)`
  - caption: The image features a Virginia opossum perched on a weathered wooden fence post...
- **PostType=photo** text=`🐕👶💖`
  - caption: The image depicts a serene outdoor scene with a baby sitting on the grass... towards a small dog lying beside them...

### User `10213660402236997 -id` — y_high=0, suicide=0.0
- **PostType=photo** text=`(empty)`
  - caption: The image features a close-up of a colorful educational toy designed to teach addition through a combination lock mechanism...
- **PostType=photo** text=`THIS NEEDS TO BE TALKED ABOUT MORE! this is horrific and I am barely seeing any mention of it`
  - caption: The image appears to be a screenshot of a tweet from Cam Lopez... discussing ICE capturing 7,000 kids...

## Llama rationales / predictions

**Not available yet.** Loading `meta-llama/Llama-3.3-70B-Instruct` requires a Hugging Face token with gated-model access (401 Unauthorized).

After exporting `HF_TOKEN` (and accepting the model license on HF), re-run:

```bash
source .venv/bin/activate
export HF_TOKEN=...   # token with access to meta-llama/Llama-3.3-70B-Instruct
PYTHONPATH=src python -m ssr.cli --config configs/real_llama_n5.yaml represent
```

Rationales will land in `artifacts/llama_n5/reps/llama_3_3_70b/<UserId>.npz` → `__meta__.gen_text` as `Rationale:` / `Prediction: RISK=0|1`.
