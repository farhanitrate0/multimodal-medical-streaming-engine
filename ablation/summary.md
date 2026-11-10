| Sampler | Backbone | Trainable params | Test AUROC | Sens @0.5 | Spec @0.5 | Pred. positive @0.5 | Sens @val-thr | Spec @val-thr | ECE |
|---|---|---|---|---|---|---|---|---|---|
| none | partial | 16.0M | 0.979 | 0.992 | 0.761 | 0.710 | 0.987 | 0.833 | 0.082 |
| balanced | partial | 16.0M | 0.983 | 0.987 | 0.816 | 0.686 | 0.972 | 0.897 | 0.066 |
| none | full | 24.6M | 0.983 | 0.992 | 0.791 | 0.699 | 0.982 | 0.880 | 0.071 |
| balanced | full | 24.6M | 0.988 | 0.992 | 0.842 | 0.679 | 0.985 | 0.906 | 0.061 |

Test prevalence: 0.625 (a constant 'pneumonia' predictor scores sens 1.0 / spec 0.0).
