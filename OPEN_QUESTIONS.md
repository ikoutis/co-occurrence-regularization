# Open Questions

A running list of empirical and theoretical questions that need follow-up
before or during paper writing.

---

## 1. Do regularized models actually learn better co-occurrence statistics?

We show that adding the co-occurrence penalty improves test accuracy on many
datasets, but we never verify the direct effect: do the trained models'
predictions actually produce a co-occurrence matrix that is closer to the
oracle (true-label) co-occurrence matrix?

**What to measure:** after training, compute
`co_matrix_pred = estimate_cooccurrence_matrix(softmax(out), edge_index, c)`
and compare it to the oracle `co_matrix_true` (built from ground-truth labels)
via, e.g., Frobenius norm or KL divergence.  Do this for baseline vs.
regularized model on a few representative datasets.

**Why it matters:** if the answer is no, the accuracy gains are due to some
indirect effect (e.g., implicit regularization, smoother predictions) rather
than the mechanism we claim.  If yes, it directly validates the stated
objective and strengthens the paper's narrative.
