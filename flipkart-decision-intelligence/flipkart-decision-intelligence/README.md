Domain: E-commerce

# Flipkart E-commerce Decision Intelligence (capstone)

All data are synthetic (produced by `generate_data.py`); nothing here reflects real Flipkart data. Money is in Indian Rupees (INR); the value column is in hundreds of INR. Every number quoted below is copied from a file in `outputs/`, and every number in those files is computed by the scripts in this repository (plain Python and NumPy; the only library routine used for a graded number besides array arithmetic and `numpy.linalg.solve` is `numpy.linalg.eigh` in Task 3.5).

## How to run

From a clean clone, in this order:

```
pip install -r requirements.txt
python generate_data.py      # prints: 800 10 170 c6f249ba693f1ebeed318d6d5e18de23
python part1.py
python part2.py
python part3.py
python part4.py
```

`common.py` is a shared helper (loading, the 560/240 split, logistic regression, KNN, metrics); importing it runs nothing. Each script regenerates its files in `outputs/`.

## Part 1 findings

**R1.1** H0 (average order value = 8.0, i.e. INR 800) is **rejected**: `z_stat = 2.8930`, which is beyond the critical value 1.96 (two-tailed, alpha = 0.05). A Type I error here would mean concluding that the true average order value differs from INR 800 when it is in fact INR 800, so the team would revise a correct plan. The sample mean is 8.9382, a gap of 0.9382 units (about INR 94, roughly 12% above the plan); the 95% interval is 8.3026 to 9.5738 (INR 830 to INR 957). The gap is statistically clear and, at about 12% of planned order value, large enough to matter for category revenue plans, although individual orders vary widely (`sd_value = 9.1723`).

**R1.2** `n_required = 1293` for a margin of error of 0.5 units (INR 50) at 95% confidence, against 800 rows available. With 800 rows the margin is 1.96 x 0.3243 = about 0.64 units (INR 64), so the current data fall short of the INR 50 target; the team would need about 493 more records.

**R1.3** `gd_slope_gap` is 0.0000 (about 3e-16): the cost J is a bowl-shaped (convex) function of the slope and intercept with a single minimum, and with a sensible step size 500 full-batch updates walk down to that minimum, which is exactly where the closed-form formula points (`gd_J100 = gd_J500 = 40.3698`, `reg_slope = 0.6701`).

**R1.4** `resid_sd_ratio_raw = 3.7525` and `resid_sd_ratio_log = 1.0396`. The raw model shows a strong funnel: residual spread in the upper half of fitted values is almost four times that in the lower half (8.6891 versus 2.3155). The log-log model removes it (0.5369 versus 0.5165, ratio close to 1).

**R1.5** Predictors with |coefficient| above 0.05 (mechanical rule, constant excluded): ln(X4) (`mr_beta_1 = 1.1576`), X2 complaint count (`mr_beta_3 = 0.0940`), X3 discount share (`mr_beta_4 = 0.1067`) and X6 cash-on-delivery flag (`mr_beta_6 = 0.1518`). This is not a ranking of importance. `mr_adj_r2 = 0.7403` against `loglog_adj_r2 = 0.6561`, so the extra predictors do earn their place (adjusted R-squared rises by about 0.084 even after the penalty for seven predictors).

## Part 2 findings

Split: `train_rows = 560` (119 positives) and `test_rows = 240` (51 positives).

**R2.1** After 200 updates the losses are, lowest to highest: eta 1.0 (0.3703), eta 0.1 (0.3776), eta 0.01 (0.5091). A learning rate that is too small has barely moved from the start after 200 updates, while on these standardised features even eta 1.0 stayed stable and converged fastest; a larger rate risks overshooting, though it did not here.

**R2.2** The three largest absolute weights are X3 discount share (0.9013), X1 tenure (-0.5554) and X2 complaints (0.5505). The largest odds ratio is `odds_ratio_X3 = 2.4629`: each rise of one training SD (0.2284) in the share of orders placed with a discount multiplies the odds that a customer churns by about 2.5, other things equal.

**R2.3** `blind_accuracy = 0.7875` (predict "stays" for everyone) versus accuracy 0.8042 at tau = 0.5: the model beats doing nothing by under two points, so accuracy is weak here because most customers do not churn and a model can score well while missing most churners (at tau = 0.5 recall is only 0.4706). Lowering tau from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634.

**R2.4** Logistic at tau = 0.3 has `f1 = 0.5714` and `recall = 0.7451`; KNN (best K = 5) has `knn_f1 = 0.3696` and `knn_recall = 0.3333`. Logistic regression is clearly better on both.

**R2.5** I would deploy the logistic model at `cost_tau = 0.15`. On the test rows it costs INR 6,050 (`cost_test_at_tau`), against INR 11,800 at tau = 0.5 (`cost_test_at_0.5`) and INR 20,400 if nobody is flagged (`cost_test_flag_nobody`). The threshold was chosen on training rows so that the test rows stay an honest, untouched check of how the choice performs on new customers.

## Part 3 findings

**R3.1** WCSS drops: 2 to 3 = 659.8282, 3 to 4 = 108.0739, 4 to 5 = 171.0575, 5 to 6 = 45.5466. The elbow rule gives `elbow_k = 3`: the drop from 3 to 4 (108.07) is less than half the preceding drop (659.83). The drops do not shrink steadily (4 to 5 is larger than 3 to 4), because the pinned starting rows give a reproducible solution, not necessarily the global optimum. K = 3 is used throughout.

**R3.2** Segments (from `part3_segments.csv`):

| Segment | Size | Business name | Profile | Action |
|---|---|---|---|---|
| A | 322 | Steady loyalists | Longest tenure (47.5 months), few complaints (1.95), low discount share (0.225), highest loyalty (62.2); outcome rate 0.0186 | Keep them with loyalty perks and early access; do not spend retention coupons on them |
| B | 255 | High-spend, complaint-prone | Highest past order value (28.7, about INR 2,870) and highest value per order (17.12), most complaints (6.07); outcome rate 0.2353 | Fix service problems first: priority support and a callback after a complaint |
| C | 223 | Bargain-focused newcomers | Shortest tenure (14.4 months), highest discount share (0.712), lowest loyalty (38.5); outcome rate 0.4664 | Targeted retention coupons and a loyalty-programme onboarding offer |

Segment C has the highest `outcome_rate`, 0.4664 (nearly half of its customers churn).

**R3.3** Single linkage cuts into 4 clusters: {E2, E3, E6}, {E1}, {E4}, {E5}. Complete linkage cuts into 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Single linkage joins clusters by their closest pair, so it chains the merges at heights 1.3, 1.4, 2.8, 2.9, 3.4, whose biggest gap (1.4 to 2.8) falls early; complete linkage uses the farthest pair, so the last merge (E1 to E2) jumps to 7.9 and the biggest gap falls at the end, giving two clear groups.

**R3.4** Step 1: the mean vector of the standardised data is all zeros (so centring changes nothing). PC1 (52.27% of the variance) contrasts tenure (+0.5317) and loyalty score (+0.4418) against discount share (-0.5187), complaints (-0.4100) and past order value (-0.2916): a loyal-and-settled versus discount-driven-and-unhappy axis. `components_for_80pct = 3` (two give 75.93%, three give 86.16%). `mean_pc1_segment_A = 1.7773`, `B = -0.7831`, `C = -1.6709`: PC1 almost reproduces the three segments, A at the loyal end and C, the high-churn segment, at the opposite end.

## Part 4 findings

**R4.1** S4 (tenure of at most 24 months) is the best split by both measures (`info_gain = 0.1252`, `gini_gain = 0.0606`) because it separates the classes most: 45.7% of the 188 short-tenure customers churn against 8.9% of the other 372, which leaves the lowest weighted impurity. A tree grown until every leaf is pure ends with tiny leaves that memorise noise in the 560 training rows, so it would overfit and do worse on new customers; limiting the maximum depth (or requiring a minimum number of rows per leaf) stops this.

**R4.2** I would ship the logistic model. At tau = 0.3 it has `f1 = 0.5714` and `recall = 0.7451`; the stump has 0.5532 and 0.7647, hard vote 0.5344 and 0.6863, soft vote 0.5507 and 0.7451, KNN 0.3696 and 0.3333. Combining models did not help: neither vote beats logistic alone on F1, and the weak KNN pulls the votes down.

**R4.3** After round 1 normalisation the two misclassified points (3 and 7) carry a total weight of 0.5000 (0.2500 each). Round 2 trusts h2 more because, on the reweighted data, h2 errs with weight 0.1667 against 0.2500 for h1 in round 1, so `round2_alpha = 0.8047` exceeds `round1_alpha = 0.5493`. The ensemble classifies 6 of 8 points correctly (`ensemble_accuracy = 0.7500`).

**R4.4** Model A option 2 (shapes from the formula (input - filter + 2 x padding) / stride + 1):

| Layer | Output shape (h x w x depth) | Parameters |
|---|---|---|
| Input | 28 x 28 x 1 | 0 |
| Conv 5x5, 6 filters, pad 2 (+ReLU) | 28 x 28 x 6 | 156 |
| Max-pool 2x2, stride 2 | 14 x 14 x 6 | 0 |
| Conv 5x5, 16 filters, pad 0 (+ReLU) | 10 x 10 x 16 | 2,416 |
| Max-pool 2x2, stride 2 | 5 x 5 x 16 | 0 |
| Flatten | 400 | 0 |
| Dense 32 (+ReLU) | 32 | 12,832 |
| Dense output 10 | 10 | 330 |

`cnn_total = 15734` against `mlp_total = 13002` for the fully connected option. For a 128 x 128 x 3 input, a fully connected layer of 1,024 neurons needs `fc_layer_params = 50332672` parameters while a convolutional layer of 64 filters of 3 x 3 needs `conv_layer_params = 1792`, because filters are shared across positions.

Model B shapes: W_xh is 128 x 64, W_hh is 128 x 128, W_hy is 16 x 128; `rnn_total = 26768` does not grow with longer sequences because the same weights are reused at every time step.

For 10 classes I would use ReLU in the hidden layers and softmax at the output with categorical cross-entropy loss. In a very deep plain network the gradient is a product of many factors below 1 as it travels backwards, so it shrinks towards zero and the early layers barely learn; a skip connection adds the input of a block to its output, giving the gradient a direct path that bypasses those shrinking factors.

## Model-choice memo

**To:** Head of Retention  **Re:** Which churn model to put into production

I recommend the logistic-regression model, flagging customers whose churn probability is at least 0.15. The threshold was chosen on training rows only, using the costs of INR 50 per unnecessary coupon and INR 400 per lost customer. On the 240 held-out customers it costs INR 6,050, against INR 11,800 at the usual 0.5 cut-off and INR 20,400 if nobody is flagged. It flags about half of customers (118 of 240) and catches 45 of the 51 who left.

Why logistic: at a 0.3 threshold its F1 is 0.5714 and recall 0.7451 in `part2_results.txt`; K-nearest neighbours reaches only 0.3696 and 0.3333. A one-question stump (tenure of 24 months or less) is close (F1 0.5532), but voting combinations did not beat logistic regression (hard vote 0.5344, soft vote 0.5507 in `part4_voting.csv`).

Risk: the evidence rests on one split with only 51 churners among the test customers, so a few customers more or less would move these costs noticeably, and the data are synthetic. Re-check the threshold on real recent data before launch.

Neither Model A (product pictures) nor Model B (review text) is worth building before the structured churn model is in production, because they answer different questions and the churn model already delivers a measurable saving.

## Limitations

1. The data are synthetic, generated by a known formula, so results show how the methods behave and do not describe real Flipkart customers, products or policies.
2. A single fixed 560/240 split was used: the best K, the threshold of 0.15 and all test costs are valid for this split only and could change with another split.
3. The test set has only 51 positive records, so recall, precision and costs have wide uncertainty (one customer changes cost by INR 50 to INR 400); no confidence intervals were computed.
4. The cost figures assume fixed costs (INR 50 per coupon, INR 400 per lost customer) and that every flagged customer gets a coupon regardless of whether it works; the real coupon effect is not modelled.
5. K-means uses pinned starting rows and may not be the global optimum, and the hierarchical-clustering table is a supplied table that is not linked to the 800 records.
