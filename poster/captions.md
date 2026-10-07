# Poster captions

Claim: LLM hidden states plus a multi-task network predict high suicide risk better than Ophir et al. (2020).

This is a research reproduction. It is not a screening tool. Scores use held-out posts. Clinical scales are used only while training.

The paper's general-risk multi-task result, 0.746, is a different label (about 36% positive) and is not plotted.

## 1. Pipeline

Facebook posts from one user pass through four language models. Each model contributes layer vectors at four positions. Attention pooling, fit on the training fold, compresses those vectors to 1024 features used by both classifiers.

## 2. Attention fusion

Inside one model and one layer, four position vectors receive scores, become weights that sum to one, and are added together. Every block is concatenated and mapped to 1024 dimensions.

## 3. Classifiers

The single-task network maps the features to a suicide score. The multi-task network predicts personality, psychosocial scores, and PHQ-9 with GAD before the suicide score. The reported multi-task score adds one tenth of a PHQ-9 value predicted from the training fold.

## 4. Cohort

1003 users. 132 are high risk, a suicide score of at least 3, about 13%. Five stratified folds, seed 42. Development chooses the model. The test fold is scored once.

## 5. High suicide risk

Paper single-task 0.629. Paper multi-task 0.697 (95% CI 0.690–0.707, Cohen's d 0.729). This single-task model 0.716. This multi-task model plus PHQ-9 0.734 (Cohen's d 0.886, same conversion as the paper).

## 6. Folds

Single-task folds 0.771, 0.657, 0.779, 0.740, 0.634 (mean 0.716). Multi-task alone 0.729, 0.713, 0.767, 0.732, 0.690 (mean 0.726). Multi-task plus PHQ-9 0.735, 0.712, 0.769, 0.742, 0.714 (mean 0.734). Dashed lines are those means. The gain is largest on folds 1 and 4. Folds 0 and 2 remain higher for the single-task model.

The plotted network is the one configuration with the best average development AUC. The PHQ-9 weight is the single shared value that raised that development score.

## 7. ROC

Each thin line is one test fold. Each thick line is the average of those five curves, and the band is ± one standard deviation across folds. The legend is the mean test AUC with its fold-to-fold spread: STM 0.716 ± 0.060, MTM + PHQ-9 0.734 ± 0.021. The multi-task model is both higher on average and steadier across folds. The dashed diagonal is chance.

## 8. System

One diagram of the path from posts to a score. Four language models emit a vector at four positions in each layer. Inside each model and layer, a bias-free linear score, a softmax, and a weighted sum pool those positions. The 14 pooled vectors are concatenated and projected to 1024 features. The single-task network passes those features through 1 to 3 tanh layers to a suicide logit; dashed layers are used on folds 0 and 2 only. The multi-task network has one shared 512-unit layer, then predicts personality, psychosocial scores, PHQ-9 and GAD, and the suicide logit in turn. Every stage after the first also receives the shared layer again. The reported multi-task score adds 0.1 times a PHQ-9 value predicted from the same features.

## 9. Hyperparameters

Single-task settings differ by fold: fold 0 is 3 layers of 1024 tanh units, learning rate 0.01, 1000 epochs; fold 1 is 1 layer of 64, learning rate 0.005, 1000 epochs; fold 2 is 3 layers of 1024, learning rate 0.005, 5000 epochs; fold 3 is 1 layer of 512, learning rate 0.001, 5000 epochs; fold 4 is 1 layer of 512, learning rate 0.05, 2500 epochs. The multi-task network is the same on every fold: 1 layer, 512 tanh units, learning rate 0.005, 1000 epochs. Both use RMSprop, momentum 0.9, batch 32, patience 200, seed 42. Fusion uses Adam, learning rate 0.01, patience 50, and a positive-class weight of 1.5 times the negative-to-positive ratio. Bars show each fold's test AUC; the higher model in each row is bold, and the dashed line marks the paper's high-risk multi-task result, 0.697.

## 10. Decisions by score

Each user is scored by the fold that held them out. Each bar is 100% of the users with that true score; the numbers inside are user counts. Green is a correct call, red an incorrect one. High risk is a true score of 3 or above, shaded on the right. The threshold maximizes F1 on that fold's development users. Thresholds are 0.525, 0.865, 0.006, 0.005, and 0.345. Miss rates: score 0 is 16% (105 of 642), score 3 is 62% (38 of 61), score 5 is 55% (18 of 33). Sensitivity is 63 of 132. Specificity is 709 of 871. Accuracy is 77%, below 87% from calling every user low risk. AUC does not use this threshold.

## 11. Scale correlations

Pearson correlations among the questionnaires and the suicide score, all 1003 users. Labels are colored by MTM stage. The strongest pairs are worry with neuroticism (0.76) and GAD with PHQ-9 (0.75). With the 0–6 suicide score (outlined bottom row): PHQ-9 0.44, brooding 0.38, GAD 0.38, loneliness 0.35, worry 0.35, neuroticism 0.30, extraversion −0.19, conscientiousness −0.18, agreeableness −0.18, openness 0.07. Satisfaction with life is omitted: its column in the data file repeats the extraversion scores (range 2–10, the BFI-10 scale), so the real life-satisfaction score is missing.

## 12. Middle-scale scores

Test correlation between each multi-task head and the matching questionnaire, grouped by cascade stage. Light dots are the five folds; the diamond and bar are the mean ± SD. The strongest means are brooding 0.13 and openness 0.13. PHQ-9 is 0.10. Worry, neuroticism, and agreeableness are about zero. The suicide gain does not come from accurate middle-scale predictions.

## 13. Examples

A confusion grid of held-out users: 63 high-risk users flagged, 69 missed, 162 false alarms, and 709 correctly cleared. Each cell shows its most extreme user, with a gauge for the model score against that fold's threshold. Quotes are verbatim excerpts from that user's posts, picked by hand as the passages most related to distress; names are removed. The correctly flagged user (true score 5) writes openly and repeatedly about suicide. The missed user (true score 5) has only a few distressed lines among 189 cheerful posts, and the model scores them near zero. The false alarm (true score 0) writes self-critical, low-mood posts but reported no ideation. The correctly cleared user (true score 0) mentions death and feelings only as jokes.

## 14. Conclusions and limitations

High-risk AUC is 0.734, compared with 0.697 for the paper's high-risk multi-task model and 0.716 for this single-task model. The comparison with the paper's 0.746 does not apply: that figure is general risk. Yes/no decisions at a development F1 threshold are weaker than a majority-class rule. The middle heads barely track the questionnaires. About 19 high-risk development users select the model. This is not a screening tool.
