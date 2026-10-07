# Decision-model probability coherence

Decision models accept a state and a runtime schema, then return probabilities
for bounded decisions. This repository asks a basic question: when several
questions describe the same underlying uncertainty, do their probabilities
describe one valid probability distribution?

The answer for the current Clef-Flash model is **not necessarily**. This
repository reproduces that problem, defines the event structures that a model
must know in order to normalize correctly, and implements a coherent
shared-utility decision head.

## The problem found in Clef-Flash

Consider one latent sentiment variable with exactly three possible values:

```text
negative | neutral | positive
```

The same uncertainty can be presented to Clef-Flash as:

```text
choice(negative, neutral, positive)
score(negative, neutral, positive)
noul(negative)
noul(neutral)
noul(positive)
```

If these questions denote the same mutually exclusive and exhaustive event,
probability theory requires

$$
P_{\text{choice}}(i)
=P_{\text{score}}(i)
=P_{\text{noul}}(i=\mathrm{true})
$$

for every label $i$, and

$$
\sum_i P_{\text{noul}}(i=\mathrm{true})=1.
$$

Clef-Flash processes the questions jointly, but returns a separately normalized
distribution for each question. Joint representation learning can encourage
agreement; it does not impose the equations above.

On 300 held-out TweetEval sentiment examples, we observed:

| Model | Mean choice–noul gap | Mean choice–score gap | Mean noul simplex error |
|---|---:|---:|---:|
| Clef-Flash original head | 0.110 | 0.036 | 0.304 |
| Coherent replacement head | 0.000 | 0.000 | $2.1\times10^{-8}$ |

For individual examples, Clef's three mutually exclusive `noul` probabilities
can sum to substantially more than one. Each answer may look reasonable in
isolation while the collection cannot be interpreted as one probability model.

The first notebook measures the issue. The second replaces Clef's decision head
while retaining the same frozen Qwen3.5-9B backbone.

## The real primitive: propositions and valid worlds

`noul`, `choice`, and `score` are useful API presentations, but they should not
be the probabilistic primitives. The primitive is:

1. a collection of propositions $A_1,\ldots,A_K$;
2. the logical relationship among them;
3. a probability distribution over the worlds allowed by that relationship.

A world $\omega$ is the subset of propositions that are true. Let
$\Omega\subseteq 2^{\{1,\ldots,K\}}$ be the set of valid worlds. A completely
general decision model assigns one utility $s_\omega(x)$ to every valid world:

$$
P(\omega\mid x)
=\frac{\exp s_\omega(x)}
       {\sum_{\omega'\in\Omega}\exp s_{\omega'}(x)}.
$$

The probability that a proposition is true is its marginal:

$$
P(A_i\mid x)
=\sum_{\omega\in\Omega:\,i\in\omega}P(\omega\mid x).
$$

Every scenario below is obtained by changing $\Omega$, or by using a
computationally cheaper factorization of this base distribution.

## Logical scenarios

### 1. Mutually exclusive and exhaustive

Exactly one proposition must be true.

```text
What is the sentiment?
negative | neutral | positive
```

The valid worlds are

$$
\Omega=\{\{1\},\{2\},\ldots,\{K\}\}.
$$

One utility $u_i$ per proposition is sufficient:

$$
p_i=P(A_i\mid x)=\frac{e^{u_i}}{\sum_j e^{u_j}}.
$$

This is the scenario currently implemented by the coherent head. All views come
from the same $p$:

$$
\begin{aligned}
\operatorname{choice}(i) &= p_i,\\
\operatorname{noul}(A_i=\mathrm{true}) &= p_i,\\
\operatorname{noul}(A_i=\mathrm{false}) &= 1-p_i.
\end{aligned}
$$

The equivalent binary log-odds form is

$$
P(A_i)=\sigma\left(u_i-operatorname{logsumexp}_{j\ne i}u_j\right).
$$

It is important that this is **not** an independent $\sigma(u_i)$. The other
propositions jointly define what “not $A_i$” means.

### 2. Mutually exclusive but not exhaustive

At most one named proposition can be true, but none may apply.

```text
What caused the symptom?
flu | allergy | bacterial infection | none of these
```

The valid worlds add the empty world:

$$
\Omega=\{\varnothing,\{1\},\ldots,\{K\}\}.
$$

Introduce a learned `none` utility $u_0$ and normalize over $K+1$ outcomes:

$$
P(A_i)=\frac{e^{u_i}}{e^{u_0}+\sum_j e^{u_j}},
\qquad
P(\mathrm{none})=\frac{e^{u_0}}{e^{u_0}+\sum_j e^{u_j}}.
$$

Consequently,

$$
\sum_iP(A_i)=1-P(\mathrm{none})\leq1.
$$

A choice that includes `none` returns this distribution directly. A forced
choice among only a subset $S$ is a conditional question:

$$
P(A_i\mid \bigvee_{j\in S}A_j)
=\frac{P(A_i)}{\sum_{j\in S}P(A_j)},
\qquad i\in S,
$$

provided the alternatives in $S$ are mutually exclusive.

### 3. Non-exclusive and non-exhaustive

Any subset, including the empty subset, may be true.

```text
What appears in the image?
contains a dog
contains a person
is outdoors
```

The image may contain both a dog and a person, or none of the named features.
Here $\Omega=2^{\{1,\ldots,K\}}$. The simplest scalable approximation is a
factorized Bernoulli model:

$$
P(A_i)=\sigma(a_i),
$$

$$
P(\omega)=
\prod_{i\in\omega}P(A_i)
\prod_{i\notin\omega}(1-P(A_i)).
$$

The probabilities do not need to sum to one across propositions. For example,
$P(\mathrm{dog})=0.9$ and $P(\mathrm{person})=0.8$ can both be valid.

In practice, the binary logit should have an explicit negative reference, such
as $a_i=u_{i,\mathrm{true}}-u_{i,\mathrm{false}}$. Utilities learned only through
a categorical softmax have an arbitrary additive offset and cannot be reused as
independent sigmoid logits without additional training.

The factorized model assumes conditional independence. When proposition
dependencies matter, the model needs interaction terms, an autoregressive
factorization, a graphical model, or explicit utilities over subsets.

### 4. Non-exclusive but exhaustive

One or more propositions must be true; the empty world is forbidden.

```text
Which systems contributed to the incident?
network | storage | application
```

Several systems may contribute, but at least one must have caused the recorded
incident. The valid worlds are

$$
\Omega=2^{\{1,\ldots,K\}}\setminus\{\varnothing\}.
$$

For small $K$, we can score all non-empty subsets directly. A cheaper model can
start with independent Bernoulli probabilities and condition on at least one
being true:

$$
P(\omega\mid\omega\ne\varnothing)
=\frac{P_{\mathrm{Bernoulli}}(\omega)}
       {1-P_{\mathrm{Bernoulli}}(\varnothing)},
\qquad\omega\ne\varnothing.
$$

This preserves the exhaustive constraint but retains the independence
assumption before conditioning.

## Why `noul`, `choice`, and `score` are derived views

Once the scenario and its canonical distribution are known, a separate neural
head for each API type is unnecessary.

### Binary view

`noul(A)` asks for an event marginal:

$$
P(\mathrm{true})=P(A),
\qquad
P(\mathrm{false})=1-P(A).
$$

How $P(A)$ is computed depends on the scenario: a categorical marginal for
exclusive alternatives, a Bernoulli marginal for independent propositions, or
a sum over valid worlds in the general case.

### Categorical or forced-choice view

`choice(A_1,\ldots,A_m)` either reports an existing categorical distribution or
conditions a canonical distribution on the listed alternatives. These meanings
must be distinguished in the schema. “Which event occurred?” and “which option
is best if forced to choose?” are not automatically the same random variable.

### Ordinal view

`score` requires mutually exclusive ordered levels with numeric values $v_i$.
It uses the same categorical probabilities and additionally returns

$$
\mathbb E[S\mid x]=\sum_i v_iP(S=i\mid x).
$$

Thus an equivalent `choice` and `score` should have the same probability vector;
ordering and expectation are metadata and post-processing, not reasons to learn
another distribution.

## Proposed relationship-aware interface

Instead of asking the model to infer probability semantics from `noul`,
`choice`, or `score`, a schema should declare the event relationship:

```python
{
    "relationship": "exclusive_exhaustive",
    "propositions": {
        "negative": "The text expresses negative sentiment.",
        "neutral": "The text expresses neutral sentiment.",
        "positive": "The text expresses positive sentiment.",
    },
    "views": [
        {"type": "categorical"},
        {"type": "binary", "proposition": "negative"},
        {"type": "ordinal", "values": [-1, 0, 1]},
    ],
}
```

The model then follows one pipeline:

```text
state + proposition descriptions
              ↓
      shared proposition utilities
              ↓
 relationship-aware probability model
              ↓
 binary / categorical / ordinal views
```

The relationship, not the output formatting, determines the mathematics.

## What the current implementation establishes

The implemented head covers the mutually-exclusive-and-exhaustive scenario. On
the same 300-example test cohort:

| Model | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Clef-Flash original head | **63.7%** | **0.873** | **0.538** | 0.187 |
| Clef-Flash with coherent replacement | 57.3% | 0.956 | 0.563 | **0.054** |

The replacement guarantees coherence and, after validation-set temperature
scaling, is better calibrated by ECE. It does not yet match Clef's accuracy.
That comparison separates three different objectives:

- **logical coherence:** enforced by the probability architecture;
- **discrimination:** learned from labeled examples;
- **calibration:** fitted and evaluated on held-out data.

The remaining research work is to implement the other relationship types and
train across heterogeneous, schema-randomized tasks so that the coherent head
retains Clef-like generalization.

## Repository layout

- `src/decision_models/base.py`: extensible Qwen-backed `DecisionModel`.
- `src/decision_models/coherent.py`: shared-utility exclusive/exhaustive head.
- `src/decision_models/clef_flash.py`: adapter for the official
  `Cloudflare/clef-flash` release and its frozen Qwen backbone.
- `src/decision_models/benchmark.py`: TweetEval transformation and coherence
  metrics.
- `src/decision_models/training.py`: embedding cache, head training, metrics,
  and validation-set temperature scaling.
- `notebooks/01_clef_flash_probability_coherence.ipynb`: reproduce the
  cross-question inconsistency.
- `notebooks/02_train_coherent_decision_head.ipynb`: compare the original and
  replacement heads on the same examples.
- `tests/`: fast CPU tests that do not download model weights.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[clef,test]'
pytest
```

Clef-Flash is a 9B BF16 model. The full notebooks require a CUDA device with
sufficient memory. Their expensive cells are controlled by environment flags,
as documented inside each notebook.
