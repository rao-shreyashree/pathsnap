# Review one — Person B

The distance-weighted KNN baseline ran on 80 training trips and 10 student-test trips from the supplied dataset. Four trip-grouped folds on the training set produced the cross-validation scores. The student-test metrics cover 19,160 GPS observations.

| Metric | Result |
|---|---:|
| Grouped CV candidate macro F1, mean ± sample SD | 0.834 ± 0.018 |
| Student-test observation top-1 accuracy | 78.12% |
| Student-test candidate macro F1 | 0.854 |
| Student-test candidate log loss | 0.475 |
| Student-test candidate miss rate | 3.56% |
| Student-test direct arc continuity | 86.52% |

Open results/comparison.png for the review chart. The fold-level values and timings are in results/knn_metrics.csv.

## What to say

“I built a common evaluation harness and tested it with distance-weighted KNN. KNN votes using 15 neighbors after scaling three geometric features. I kept trips together in cross-validation and evaluated on all ten student-test trips. It picked the aligned reference arc first for 78.12% of GPS observations. In 3.56% of cases, the reference arc was missing from the candidate set, so a classifier could not recover it. The next team comparison will use one agreed feature and label pipeline across all five models.”

## What the results measure

These figures use the included temporary candidate generator and order-preserving GPS-to-route alignment. Pointwise labels are estimates derived from the provided route sequence. The direct-continuity metric checks consecutive winning arcs for equality or edge-to-edge contact and can miss valid travel through intermediate arcs. Evaluator trips were excluded throughout. The evaluator UI belongs to the later integration phase.
