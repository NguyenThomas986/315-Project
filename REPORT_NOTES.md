# Report Notes — Twitch Streamer Viewership Prediction

A reference for writing the final report. Contains a summary of what the pipeline does,
the actual results from the training run, and guidance on what each report section should cover.

---

## What the Pipeline Does (End-to-End Summary)

1. **Load & filter** — Reads `topstreamers.csv` (999 rows). Removes 57 esports accounts,
   leaving 942 individual/personality streamers.

2. **Preprocess** — Categorical features (language, games, active day, etc.) are
   frequency-encoded (replaced with their relative frequency in the dataset).
   All 14 features are z-score normalized.

3. **K-means clustering** — Runs K-means for k=2 through k=10. Evaluates each k using the
   Dunn Index. k=2 scored highest (0.1231), so the data is split into two clusters.

4. **Cluster profiling** — Computes per-cluster averages on key metrics to identify which
   cluster represents high-viewership streamers.

5. **Classification** — A PyTorch MLP is trained to predict which cluster a streamer belongs
   to, evaluated with 5-fold stratified cross-validation.

---

## Actual Results from the Run

### Clustering
| k | Dunn Index | Inertia |
|---|---|---|
| 2 | **0.1231** | 11,452.5 |
| 3 | 0.0883 | 10,446.9 |
| 4 | 0.0883 | 9,564.8 |
| 5 | 0.0904 | 8,893.4 |
| ... | ... | ... |

**Optimal k = 2**

### Cluster Profiles
| Metric | Cluster 0 (176 streamers) | Cluster 1 (766 streamers) |
|---|---|---|
| Avg viewers/stream | 41,284 | 8,673 |
| Followers gained/stream | 3,376 | 3,347 |
| Avg stream duration (hrs) | 7.2 | 5.6 |
| Active days/week | 5.4 | 3.3 |
| Total followers | 2,492,318 | 546,984 |
| Total views | 101,718,017 | 10,072,511 |
| Top language | English (44%) | English (37%) |
| Top game | Just Chatting (38%) | Just Chatting (25%) |
| Most active day | Wednesday (20%) | Tuesday (19%) |

**Cluster 0 = high-viewership group.**

### Classification (MLP: hidden=[256,128,64], lr=0.0005, dropout=0.3)
| Fold | Accuracy |
|---|---|
| 1 | 98.94% |
| 2 | 97.35% |
| 3 | 96.81% |
| 4 | 95.74% |
| 5 | 97.87% |
| **Mean ± Std** | **97.34% ± 1.07%** |

Target was ≥ 70%. Achieved.

---

## Report Section Guidance

### 1. Introduction
- Motivate with Twitch's growth (especially post-COVID surge) and the challenge new streamers
  face in growing an audience in a crowded platform.
- State the core question: *What combination of streaming habits maximizes viewership?*
- Briefly mention the approach: filter to individual streamers, cluster by behavior,
  train a classifier to assign new streamers to the best-fit group.
- Summarize the result: two distinct streamer archetypes emerged; the classifier achieves
  97.34% accuracy in assigning streamers to their cluster.

### 2. Data Mining Task
- **Input:** Behavioral and categorical data for 942 personality streamers
  (stream duration, active days, games played, language, follower counts, etc.).
- **Output:** A cluster label (0 or 1) identifying which streamer archetype a creator
  belongs to, and which archetype maximizes viewership.
- **Questions to list:**
  - Do natural groupings of streamers exist based on their habits?
  - What habits distinguish high-viewership streamers from lower-viewership ones?
  - Can a model reliably assign new streamers to a cluster?
- **Challenges to mention:**
  - Small dataset (942 rows after filtering) — limited generalizability.
  - No engagement data (chat, subscriptions) — viewer count used as proxy.
  - Dataset reflects only the top 1000 streamers; findings may not apply to smaller creators.

### 3. Technical Approach
- **Preprocessing:** Explain frequency encoding for categoricals and z-score normalization.
  Note why z-score was chosen (K-means is distance-based, so scale matters).
- **K-means:** Describe the algorithm (assign → compute centroid → reassign, repeat).
  Explain Dunn Index: measures ratio of minimum inter-cluster distance to maximum
  intra-cluster diameter — higher means tighter, better-separated clusters.
  Include the Dunn Index plot (`results/plots/dunn_index.png`).
- **MLP Classifier:** Describe the architecture — Linear → BatchNorm → ReLU → Dropout,
  repeated for each hidden layer, ending in a softmax output over 2 classes.
  Mention CrossEntropyLoss, Adam optimizer, dropout for regularization.
- **Cross-validation:** Explain stratified 5-fold CV and why it's used
  (prevents overfitting evaluation on small datasets).
- A block diagram showing the full pipeline (filter → preprocess → cluster → classify)
  would satisfy the pseudocode/figure requirement.

### 4. Evaluation Methodology
- **Dataset source:** Kaggle — "Top 1000 Twitch Streamers Data" (May 2024).
- **Challenges with this data:** Outdated (2024), top-only bias, missing engagement metrics.
- **Metrics used:**
  - *Dunn Index* — to select optimal k for clustering.
  - *Classification accuracy* — to evaluate the MLP (target ≥ 70%, achieved 97.34%).
  - *Confusion matrix* — to inspect per-cluster misclassification
    (`results/plots/confusion_matrix_fold1.png`).
  - *Cross-validation std* — to confirm the model is consistent across folds (±1.07%).

### 5. Results and Discussion
- Lead with the Dunn Index plot showing k=2 is optimal. Explain what that means:
  the data naturally separates into two streamer archetypes.
- Present the cluster profile table. Tell the story clearly:
  Cluster 0 streamers average 41K viewers/stream vs. 8.7K for Cluster 1.
  They stream ~1.6 hrs longer per session and are active 2 more days per week.
  Both clusters favor Just Chatting, but Cluster 0 does so more heavily (38% vs. 25%).
- Present the fold-by-fold accuracy table and the mean ± std.
  Note that 97.34% accuracy suggests the two clusters are very well-separated —
  the model can reliably identify which group a streamer belongs to.
- **What worked:** K-means found a clean separation. The MLP learned cluster boundaries
  quickly (high accuracy even at low epochs). Z-score normalization was essential
  given the scale differences between features (e.g., total views vs. active days/week).
- **What didn't work / limitations:** k=2 means the model gives a binary answer —
  a streamer is either in the high-viewership group or not. This is a coarse recommendation.
  The dataset is small and top-heavy, so the model likely cannot generalize to
  new or smaller creators. Frequency encoding for games loses specificity
  (many games appear rarely).
- Reference the cluster profiles chart (`results/plots/cluster_profiles.png`)
  and loss curve (`results/plots/loss_curve_fold1.png`) as figures in the report.
