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

- `1:input_only`: shape=(1024,) mean=-0.0001 std=0.1169
- `1:last_prompt`: shape=(1024,) mean=0.0015 std=0.2083
- `1:cot`: shape=(1024,) mean=-0.0000 std=0.1232
- `1:final_pred`: shape=(1024,) mean=0.0167 std=0.2424
- `10:input_only`: shape=(1024,) mean=0.0131 std=0.6198
- `10:last_prompt`: shape=(1024,) mean=0.0242 std=0.7889
- `10:cot`: shape=(1024,) mean=0.0119 std=0.6550
- `10:final_pred`: shape=(1024,) mean=0.0241 std=0.9166

### Model `smollm2_360m`

- `1:input_only`: shape=(960,) mean=-0.0456 std=1.3273
- `1:last_prompt`: shape=(960,) mean=-0.0756 std=1.5657
- `1:cot`: shape=(960,) mean=-0.0593 std=1.3459
- `1:final_pred`: shape=(960,) mean=-0.0345 std=1.8481
- `11:input_only`: shape=(960,) mean=-0.0829 std=4.5117
- `11:last_prompt`: shape=(960,) mean=-0.2575 std=4.2642
- `11:cot`: shape=(960,) mean=-0.0959 std=4.3830
- `11:final_pred`: shape=(960,) mean=-0.0625 std=8.1065

## 4. Fused 1024-d vector (fit on all POC users for illustration)

k_per_block=29 n_blocks=32 out_dim=1024

vector[:16] = `[-3.0741000175476074, 0.5492134094238281, 7.270153522491455, 4.686683177947998, -6.906124114990234, 1.985206961631775, 4.904317378997803, -3.808490753173828, 0.3050362765789032, 5.745912075042725, 11.04626178741455, 5.053055763244629, 4.647622108459473, 0.6936990022659302, -6.092545509338379, -1.478501319885254]`

vector mean=0.0257 std=7.2831 l2=233.0622

## 5. CV metrics summary

| Variant | AUC-ROC | PR-AUC | F1 | Cohen's d |
|---------|---------|--------|----|-----------|
| stm_high | 0.560 | 0.447 | 0.280 | 1.178 |
| mtm_high | 0.560 | 0.530 | 0.100 | 2.141 |
