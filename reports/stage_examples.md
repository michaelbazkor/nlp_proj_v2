# Stage examples (POC run)

## 0. Cohort row

```
UserId          204720893660851 -id
suicide                         2.0
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
  image_key: `204720893660851_207722916693982 -id`
- **PostType=photo** has_image=True
  text: `Bye 🏃‍♀️🏃‍♀️`
  image_key: `204720893660851_204158203717120 -id`
- **PostType=video** has_image=True
  text: `🤣🤣🤣🤣🤣`
  image_key: `204720893660851_202474257218848 -id`

## 2. Assembled corpus (truncated)

n_posts=25 n_images=0 n_chars=1110

```
[post] It was like I never left 😡😡 waste of time [post] I do better alone ✌ [post] 🤣🤣🤣🤣 [post] My patient finally came home today 😊 Still tired tho 😐 [post] Alright I work this weekend I dont wanna see nobody I know in the ER this weekend...stay home and take Tylenol 😂😂😂 [post] It feels good knowing you're  appreciated 😍😍 [post] 😎😎 [post] Pepper trying to move the pillow to cover up his mess lol, he just put the sock there when i walked in [post] 🤣🤣🤣🤣🤣 [post] Say what now!! 🤔🤔 [post] Once I pop that pill deuces ✌😭😭😂😂 [post] Yes I did and he loves them lol 😂😂😂 Yall know these my churn 😍😍 [post] All ima say is stay off them drugs 👀👀 [post] Bye 🏃‍♀️🏃‍♀️ [post] 😂😂😂😂 [post] 😂😂😂😂 [post] Like chill mother nature i was just playing 😎😑 [post] This heffa 😂😂😂💀💀💀 [post] Soo yall really gonna share chapstick eww 😖 no ma'am. [post] It's too early for this foolishness 😑😑 [post] Just mind ya business [post] 😂😂😂😂 [post] Sooo how Spokeo just ASSUME I'm single smh,  I might have situationship 😂😂😂 and apparently I'm Caucasian and didn't even know why yall didn't tell me better [post] 🤣🤣🤣 [post] Can you just...😄😂😂
```

## 3. Raw representation blocks

### Model `qwen3_0_6b`

- `1:input_only`: shape=(1024,) mean=-0.0002 std=0.1163
- `1:last_prompt`: shape=(1024,) mean=0.0033 std=0.2210
- `1:cot`: shape=(1024,) mean=-0.0003 std=0.1210
- `1:final_pred`: shape=(1024,) mean=-0.0021 std=0.2590
- `10:input_only`: shape=(1024,) mean=0.0120 std=0.6161
- `10:last_prompt`: shape=(1024,) mean=0.0228 std=0.7454
- `10:cot`: shape=(1024,) mean=0.0114 std=0.6420
- `10:final_pred`: shape=(1024,) mean=-0.0332 std=0.9643

### Model `smollm2_360m`

- `1:input_only`: shape=(960,) mean=-0.0464 std=1.3322
- `1:last_prompt`: shape=(960,) mean=-0.0768 std=1.6019
- `1:cot`: shape=(960,) mean=-0.0633 std=1.4175
- `1:final_pred`: shape=(960,) mean=-0.0204 std=1.8878
- `11:input_only`: shape=(960,) mean=-0.0895 std=4.4381
- `11:last_prompt`: shape=(960,) mean=-0.2525 std=4.1970
- `11:cot`: shape=(960,) mean=-0.1190 std=4.1377
- `11:final_pred`: shape=(960,) mean=-0.1565 std=7.3772

## 4. Fused 1024-d vector (fit on all POC users for illustration)

k_per_block=29 n_blocks=32 out_dim=1024

vector[:16] = `[4.600985527038574, -2.510861396789551, 15.906209945678711, 5.410788059234619, -2.823784589767456, -8.831552505493164, 5.3534040451049805, -6.751702308654785, -0.5116231441497803, 1.3828368186950684, -1.7290759086608887, 1.5013114213943481, -5.162143707275391, -0.5264931917190552, 0.0784616470336914, -0.5954151749610901]`

vector mean=-0.0677 std=6.9692 l2=223.0239

## 5. CV metrics summary

| Variant | AUC-ROC | PR-AUC | F1 | Cohen's d |
|---------|---------|--------|----|-----------|
| stm_high | 0.520 | 0.440 | 0.233 | 0.072 |
| mtm_high | 0.520 | 0.417 | 0.213 | 0.072 |
