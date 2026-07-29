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

n_posts=25 n_images=0 n_chars=1110

```
[post] It was like I never left 😡😡 waste of time [post] I do better alone ✌ [post] 🤣🤣🤣🤣 [post] My patient finally came home today 😊 Still tired tho 😐 [post] Alright I work this weekend I dont wanna see nobody I know in the ER this weekend...stay home and take Tylenol 😂😂😂 [post] It feels good knowing you're  appreciated 😍😍 [post] 😎😎 [post] Pepper trying to move the pillow to cover up his mess lol, he just put the sock there when i walked in [post] 🤣🤣🤣🤣🤣 [post] Say what now!! 🤔🤔 [post] Once I pop that pill deuces ✌😭😭😂😂 [post] Yes I did and he loves them lol 😂😂😂 Yall know these my churn 😍😍 [post] All ima say is stay off them drugs 👀👀 [post] Bye 🏃‍♀️🏃‍♀️ [post] 😂😂😂😂 [post] 😂😂😂😂 [post] Like chill mother nature i was just playing 😎😑 [post] This heffa 😂😂😂💀💀💀 [post] Soo yall really gonna share chapstick eww 😖 no ma'am. [post] It's too early for this foolishness 😑😑 [post] Just mind ya business [post] 😂😂😂😂 [post] Sooo how Spokeo just ASSUME I'm single smh,  I might have situationship 😂😂😂 and apparently I'm Caucasian and didn't even know why yall didn't tell me better [post] 🤣🤣🤣 [post] Can you just...😄😂😂
```

## 3. Raw representation blocks

### Model `qwen3_0_6b`

- `1:input_only`: shape=(1024,) mean=-0.0001 std=0.1170
- `1:last_prompt`: shape=(1024,) mean=0.0035 std=0.2233
- `1:cot`: shape=(1024,) mean=-0.0005 std=0.1243
- `1:final_pred`: shape=(1024,) mean=0.0005 std=0.1944
- `10:input_only`: shape=(1024,) mean=0.0115 std=0.6159
- `10:last_prompt`: shape=(1024,) mean=0.0213 std=0.7607
- `10:cot`: shape=(1024,) mean=0.0101 std=0.6552
- `10:final_pred`: shape=(1024,) mean=0.0347 std=0.7864

### Model `smollm2_360m`

- `1:input_only`: shape=(960,) mean=-0.0471 std=1.3347
- `1:last_prompt`: shape=(960,) mean=-0.0768 std=1.6067
- `1:cot`: shape=(960,) mean=-0.0876 std=1.5748
- `1:final_pred`: shape=(960,) mean=-0.0561 std=1.8840
- `11:input_only`: shape=(960,) mean=-0.0915 std=4.4637
- `11:last_prompt`: shape=(960,) mean=-0.2522 std=4.1952
- `11:cot`: shape=(960,) mean=-0.1622 std=4.1822
- `11:final_pred`: shape=(960,) mean=-0.0785 std=5.1432

## 4. Fused 1024-d vector (fit on all POC users for illustration)

k_per_block=29 n_blocks=32 out_dim=1024

vector[:16] = `[-13.291420936584473, -16.443946838378906, 9.875502586364746, 6.203873157501221, -33.521915435791016, -6.1000189781188965, 5.328449249267578, -0.15693432092666626, -7.019228458404541, 4.398092269897461, 1.7911251783370972, -2.29988431930542, -1.9567790031433105, -1.5888131856918335, 1.3945190906524658, 1.123441219329834]`

vector mean=-0.1713 std=6.7456 l2=215.9279

## 5. CV metrics summary

| Variant | AUC-ROC | PR-AUC | F1 | Cohen's d |
|---------|---------|--------|----|-----------|
| stm_general | 0.506 | 0.543 | 0.393 | 0.036 |
| stm_high | 0.550 | 0.475 | 0.267 | 1.383 |
| mtm_general | 0.450 | 0.543 | 0.447 | -1.154 |
| mtm_high | 0.738 | 0.604 | 0.080 | 2.098 |
