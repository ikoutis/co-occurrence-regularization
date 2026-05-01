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

## 2. Can a model recover a labeling from co-occurrence statistics alone?

A cleaner version of Q1: define the *co-occurrence recovery problem* as a
pure optimization problem with no cross-entropy term.

**Setup:** generate a Stochastic Block Model graph (labels = block membership).
Compute C* from **all** true labels.  Train GCN / GT / MLP with only the
co-occurrence loss:

```
loss = edge_loss(softmax(f(G)), edge_index, -log(C* + ε))
```

No label supervision at all.  Evaluate:
- Frobenius error ||C_pred − C*||_F  (did the network satisfy the objective?)
- Clustering accuracy after Hungarian matching  (did it find the true labeling?)

**Key comparisons:**
- GCN vs GT vs MLP across homophilic and heterophilic SBM
- GT here is pure all-pairs self-attention with no positional encoding and
  no access to edge_index — the topology-free baseline

**The central hypothesis this tests:** GTs cannot recover co-occurrence
statistics on their own because they have no topology induction bias — they
treat all node pairs equally, so the edge-level constraint provides no
gradient signal that distinguishes neighbors from non-neighbors.  GCNs, by
contrast, aggregate over actual edges and can therefore feel the co-occurrence
pressure directly through message passing.  This asymmetry is precisely why
the explicit penalty helps GTs more than GNNs in the main experiments: GTs
*need* the regularization to learn what GCNs can discover on their own.

**Expected outcomes:**
- GCN: low Fro error, high clustering accuracy (especially homo)
- GT:  low Fro error possible (it can find *a* valid soft assignment via
  attention collapse), but low clustering accuracy — degenerate solution
- MLP: likely fails both (no structural signal whatsoever)

A GT achieving low Fro error but low accuracy would be the most interesting
result: it shows the GT satisfies the aggregate statistics through a uniform
or permuted assignment rather than the correct one, confirming that topology
is what breaks the symmetry.

**Script:** `experiments/cooc_recovery/cooc_recovery.py`
