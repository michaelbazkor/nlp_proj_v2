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

Each thin line is one test fold. Each thick line is the average of those five curves. The legend is the mean test AUC, 0.716 and 0.734. The gray diagonal is chance.
