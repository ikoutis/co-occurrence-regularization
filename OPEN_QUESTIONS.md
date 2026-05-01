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

---

## 2. Can a GNN recover a labeling from co-occurrence statistics alone?

A cleaner version of Q1: define the *co-occurrence recovery problem* as a
pure optimization problem with no cross-entropy term.

**Setup:** generate a Stochastic Block Model graph (labels = block membership).
Compute C* from **all** true labels.  Train a GNN / MLP with only the
co-occurrence loss:

```
loss = edge_loss(softmax(f(G)), edge_index, -log(C* + ε))
```

No label supervision at all.  Evaluate:
- Frobenius error ||C_pred − C*||_F  (did the network satisfy the objective?)
- Clustering accuracy after Hungarian matching  (did it find the true labeling?)

**Key comparisons:**
- GCN vs MLP: does neighborhood aggregation help match co-occurrence stats?
- Homophilic vs heterophilic SBM: does graph topology contain enough signal?

**Why it matters:** if a GNN can recover the labeling purely from co-occurrence
pressure, it means the penalty loss contains the full structural information
needed for classification.  If only GCN succeeds (not MLP), it confirms that
topology is essential — the model must propagate information across edges to
satisfy the edge-level constraint.  A degenerate solution (low Fro error but
low accuracy) would suggest co-occurrence statistics alone are insufficient to
identify node labels uniquely.

**Script:** `experiments/cooc_recovery/cooc_recovery.py`
