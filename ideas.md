# Ideas: constrained probabilistic routing for decision models

## 1. Motivation

Decision models such as Clef accept a state and a schema containing several
bounded questions. The model can process those questions jointly, but each
question is ultimately normalized separately. Joint representations allow one
field to influence another; they do not ensure that probabilities attached to
logically related fields obey the same probability model.

This distinction matters whenever several questions are views of the same
uncertainty or when business rules restrict valid combinations of decisions.

For example, suppose sentiment is exactly one of `negative`, `neutral`, or
`positive`. The following outputs cannot all describe the same random variable:

```text
choice:
  negative = 0.20
  neutral  = 0.75
  positive = 0.05

noul(negative) = 0.70
noul(neutral)  = 0.65
noul(positive) = 0.10
```

The categorical probabilities sum to one, but the binary probabilities sum to
1.45 and disagree with the categorical marginals.

The experiments in this repository show that this is not merely hypothetical.
On 300 held-out TweetEval sentiment examples, the original Clef-Flash head had:

- mean choice–noul probability gap: `0.110`;
- mean choice–score probability gap: `0.036`;
- mean error in the sum of exhaustive noul probabilities: `0.304`.

A replacement head derived every view from one shared distribution and reduced
these errors to floating-point noise.

The broader research objective is therefore:

> Given a state, runtime propositions, and declared logical constraints, produce
> calibrated probabilities over valid complete decisions and derive every
> user-facing output from that canonical probability model.

## 2. Separate four different concepts

The design should keep the following concepts distinct.

### 2.1 Representation interaction

Questions or fields attend to one another while computing features. This can
help predictions, but it imposes no probability identities.

### 2.2 Logical coherence

Impossible combinations receive probability zero, and equivalent events have
equal probabilities. This should be guaranteed by the probability architecture
or constrained inference.

### 2.3 Predictive discrimination

The model assigns greater probability to correct outcomes than incorrect ones.
This is measured with accuracy, ranking metrics, NLL, and Brier score.

### 2.4 Calibration

Among predictions assigned probability (p), the relevant event occurs at
approximately frequency (p). Calibration is evaluated after all constraint
processing, because normalization or projection can change confidence.

A model may be coherent but wrong, accurate but incoherent, or coherent and
accurate but miscalibrated. These properties require separate measurements.

## 3. General probabilistic formulation

Let (A_1,\ldots,A_K) be propositions. A complete world

\[
\omega\subseteq\{1,\ldots,K\}
\]

records which propositions are true. Logical rules define the allowed set

\[
\mathcal C\subseteq 2^{\{1,\ldots,K\}}.
\]

The decision model assigns an energy or utility (E_\theta(x,\omega)) to each
valid world and normalizes only over valid worlds:

\[
P_\theta(\omega\mid x)
=
\frac{\exp E_\theta(x,\omega)}
{\sum_{\omega'\in\mathcal C}\exp E_\theta(x,\omega')},
\qquad \omega\in\mathcal C.
\]

Invalid worlds can equivalently be assigned energy (-\infty).

The probability that proposition (A_i) is true is its marginal:

\[
P(A_i\mid x)
=
\sum_{\omega\in\mathcal C:\,i\in\omega}P(\omega\mid x).
\]

This formulation is the base case. Binary, categorical, ordinal, and routing
outputs are queries against the same joint distribution rather than separate
neural heads.

## 4. Fundamental event structures

### 4.1 Exclusive and exhaustive

Exactly one proposition is true.

```text
sentiment = negative | neutral | positive
```

The valid worlds are the (K) singleton sets. One utility (u_i) per
proposition produces:

\[
P(A_i)=\frac{e^{u_i}}{\sum_j e^{u_j}}.
\]

Consequences:

\[
\sum_iP(A_i)=1
\]

and

\[
P(A_i)
=
\sigma\left(u_i-\operatorname{logsumexp}_{j\ne i}u_j\right).
\]

This is the scenario implemented by the current shared-utility head.

### 4.2 Exclusive but not exhaustive

At most one named proposition is true, but none may apply.

```text
cause = flu | allergy | infection | none-of-these
```

Add an explicit empty-world utility (u_0):

\[
P(A_i)=\frac{e^{u_i}}{e^{u_0}+\sum_j e^{u_j}},
\qquad
P(\varnothing)=\frac{e^{u_0}}{e^{u_0}+\sum_j e^{u_j}}.
\]

The named probabilities satisfy

\[
\sum_iP(A_i)=1-P(\varnothing)\leq1.
\]

The `none` outcome should be modeled explicitly rather than inferred from low
confidence after an ordinary softmax.

### 4.3 Non-exclusive and non-exhaustive

Any subset, including the empty subset, may be true.

```text
contains_dog
contains_person
is_outdoors
```

A scalable baseline uses a separate Bernoulli log-odds value for every event:

\[
P(A_i)=\sigma(a_i),
\qquad
a_i=u_{i,\mathrm{true}}-u_{i,\mathrm{false}}.
\]

If events are conditionally independent,

\[
P(\omega)
=
\prod_{i\in\omega}P(A_i)
\prod_{i\notin\omega}(1-P(A_i)).
\]

Probabilities across propositions need not sum to one. A bare sigmoid applied to
utilities trained only through categorical softmax is not sufficient: softmax
utilities are invariant to a shared additive offset, while independent binary
probabilities require an absolute true-versus-false reference.

When proposition dependencies matter, replace the factorized Bernoulli model
with interaction potentials, a graphical model, an autoregressive
factorization, or explicit subset utilities.

### 4.4 Non-exclusive but exhaustive

One or more propositions must be true, while several may be true.

```text
incident_contributors = {network, storage, application}
```

The valid support excludes only the empty world:

\[
\mathcal C=2^{\{1,\ldots,K\}}\setminus\{\varnothing\}.
\]

For small (K), score all non-empty subsets. A cheaper approximation begins
with factorized Bernoulli probabilities and conditions on at least one event:

\[
P(\omega\mid\omega\ne\varnothing)
=
\frac{P_{\mathrm{Bernoulli}}(\omega)}
{1-P_{\mathrm{Bernoulli}}(\varnothing)},
\qquad \omega\ne\varnothing.
\]

This enforces exhaustiveness but does not remove the underlying independence
assumption.

## 5. Output types should be derived views

The terms `noul`, `choice`, and `score` describe how users query or display a
probability model. They should not determine separate probability models.

### 5.1 Binary view

For any event (A):

\[
P(\mathrm{true})=P(A),
\qquad
P(\mathrm{false})=1-P(A).
\]

The relationship model determines how (P(A)) is obtained.

### 5.2 Categorical view

If alternatives are already exclusive and exhaustive, return their canonical
distribution. If the user forces a choice among a subset (S) of mutually
exclusive events, return the conditional distribution:

\[
P(A_i\mid\bigvee_{j\in S}A_j)
=
\frac{P(A_i)}{\sum_{j\in S}P(A_j)},
\qquad i\in S.
\]

An unconditional categorical event and a forced preference among options are
different semantics and must be declared separately.

### 5.3 Ordinal view

For mutually exclusive ordered levels with values (v_i), use the canonical
categorical probabilities and return their expectation:

\[
\mathbb E[S\mid x]=\sum_i v_iP(S=i\mid x).
\]

An equivalent ordinal and categorical query should therefore return the same
probability vector. Ordering and expectation are metadata and post-processing.

## 6. Richer routing restrictions

The valid-world formulation supports constraints beyond the four basic event
structures.

### 6.1 Exactly one

```text
exactly_one(team)
```

\[
\sum_i y_i=1.
\]

### 6.2 At most one

```text
at_most_one(primary_cause)
```

\[
\sum_i y_i\leq1.
\]

### 6.3 At least one

```text
at_least_one(contributing_system)
```

\[
\sum_i y_i\geq1.
\]

### 6.4 Cardinality and budget

```text
at_most_k(actions, 2)
exactly_k(reviewers, 3)
```

\[
L\leq\sum_i y_i\leq U.
\]

Applications include tool selection, review allocation, and interventions
under a resource budget.

### 6.5 Implication

```text
security_incident -> escalate
```

For Boolean variables:

\[
y_{\mathrm{security}}\leq y_{\mathrm{escalate}}.
\]

The joint restriction implies the marginal inequality

\[
P(\mathrm{security})\leq P(\mathrm{escalate}).
\]

### 6.6 Mutual exclusion across fields

```text
refund -> not chargeback
```

\[
y_{\mathrm{refund}}+y_{\mathrm{chargeback}}\leq1.
\]

### 6.7 Hierarchical constraints

```text
database_failure -> infrastructure_incident
```

The child event must imply its parent, yielding

\[
P(\mathrm{database\ failure})
\leq
P(\mathrm{infrastructure\ incident}).
\]

### 6.8 Conditional option availability

```text
if team == security:
    action in {investigate, isolate, escalate}
```

Options may be masked or activated based on another decision.

### 6.9 Ordinal monotonicity

For cumulative ordinal probabilities:

\[
P(S\geq k+1)\leq P(S\geq k).
\]

A cumulative-link parameterization enforces this automatically.

### 6.10 Conservation and flow

Workflow routing may require

\[
\text{incoming flow}=\text{outgoing flow}
\]

or rules such as:

```text
if rejected:
    route to exactly one of {manual_review, customer, archive}
```

These can often be represented as linear constraints.

## 7. Example: constrained incident routing

Consider four linked fields:

```text
team:       billing | infrastructure | security
severity:   low | medium | high
page:       true | false
escalate:   true | false
```

Rules:

```text
team=security -> escalate=true
severity=high -> page=true
severity=low  -> page=false
page=true     -> escalate=true
```

Independently normalized heads could report:

```text
P(team=security) = 0.90
P(escalate=false) = 0.80
P(severity=high) = 0.85
P(page=false) = 0.75
```

With constrained routing, the model scores only valid complete records. Any
world containing `team=security` and `escalate=false` has probability zero.
Reported field probabilities are marginals of the valid joint distribution.

This guarantees, among other consequences,

\[
P(\mathrm{security})\leq P(\mathrm{escalate})
\]

and

\[
P(\mathrm{high})\leq P(\mathrm{page}).
\]

## 8. Enforcement strategies

### 8.1 Constrained parameterization

Design the output transformation so invalid probabilities cannot be expressed.

Examples:

- softmax for exclusive and exhaustive alternatives;
- softmax with an explicit `none` state;
- paired binary logits for independent events;
- cumulative-link models for ordinal outcomes;
- tree softmax for hierarchical labels;
- normalized flow networks for routing.

This is usually the simplest and most reliable strategy when the constraint
family is known in advance.

### 8.2 Normalize over valid configurations

Define an energy for a complete record:

\[
E_\theta(x,y)
=
\sum_i u_i(x,y_i)
+
\sum_{i,j}v_{ij}(x,y_i,y_j),
\]

then normalize over (y\in\mathcal C). Unary terms represent evidence for
individual fields; interaction terms represent compatibility between fields.

This corresponds to structured prediction with conditional random fields,
factor graphs, or energy-based models. Exact inference may use:

- enumeration for small state spaces;
- dynamic programming for chain or tree structure;
- variable elimination;
- belief propagation;
- probabilistic circuits;
- weighted model counting for Boolean constraints.

Approximate inference may use sampling, variational methods, or loopy belief
propagation.

The number of complete configurations may grow exponentially. Finding the
single best valid decision can be handled with SAT, integer programming, or
dynamic programming, but calibrated probabilities require a partition function
and marginals, not only the MAP assignment.

### 8.3 Project raw probabilities onto a feasible set

Given raw output (q), find the closest valid distribution:

\[
p^*
=
\arg\min_{p\in\mathcal P}D_{\mathrm{KL}}(p\|q),
\]

subject to constraints such as

\[
Ap=b,
\qquad
Cp\leq d,
\qquad
p\geq0.
\]

Possible implementations include convex optimization, iterative proportional
fitting, Dykstra-style projections, differentiable optimization layers, and
specialized Sinkhorn-like procedures.

Projection is attractive when adapting an existing model such as Clef. Its
limitation is identifiability: separately reported marginals do not uniquely
determine a joint distribution. Projection can make marginals feasible, but it
cannot recover missing correlations without extra assumptions.

### 8.4 Autoregressive constrained routing

Generate fields sequentially and mask choices that would make the partial
record impossible. This avoids enumerating all complete worlds and guarantees a
valid final record.

Trade-offs:

- probabilities depend on field order unless the model is carefully designed;
- obtaining arbitrary marginals may require summing over many paths;
- it loses the one-pass parallel scoring advantage of decision models.

### 8.5 Soft penalties or reinforcement learning

Add differentiable penalties such as

\[
L_{\mathrm{implication}}
=
\left[\max(0,p_{\mathrm{security}}-p_{\mathrm{escalate}})\right]^2
\]

or cross-view consistency penalties such as

\[
L_{\mathrm{consistency}}
=
\sum_i
\left(p_{\mathrm{choice},i}-p_{\mathrm{binary},i}\right)^2.
\]

RL can reward valid records, partial correctness, cost-sensitive decisions, and
global utility. These methods encourage compliance but do not make violations
impossible on unseen inputs.

Use hard constraints for invariants and soft objectives for preferences,
uncertain rules, or graded costs.

## 9. Calibration under constraints

Constraints and calibration interact. Renormalization, conditioning, or
projection changes probabilities, so calibration must be evaluated on the final
constrained output.

Questions to test:

1. Does temperature scaling before constrained inference remain calibrated
   afterward?
2. Is a single global temperature sufficient, or are relation-specific
   temperatures needed?
3. Does projecting calibrated marginals onto a feasible polytope damage
   marginal or joint calibration?
4. Should calibration minimize marginal Brier score, joint NLL, or expected
   downstream decision cost?
5. Can conformal methods provide coverage guarantees over valid structured
   outputs?

The current experiment demonstrates that temperature scaling can improve ECE
while preserving exact shared-softmax coherence. More complex constraints need
their own post-inference calibration study.

## 10. Proposed schema and API

The schema should declare variables and relationships separately from requested
views.

```python
ConstraintSpec(
    variables={
        "team": Categorical(["billing", "infrastructure", "security"]),
        "severity": Ordinal(["low", "medium", "high"]),
        "page": Boolean(),
        "escalate": Boolean(),
    },
    constraints=[
        Implies(Eq("team", "security"), Eq("escalate", True)),
        Implies(Eq("severity", "high"), Eq("page", True)),
        Implies(Eq("severity", "low"), Eq("page", False)),
        Implies(Eq("page", True), Eq("escalate", True)),
    ],
    views=[
        Marginal("team"),
        BinaryEvent(Eq("team", "security")),
        ExpectedValue("severity", values=[0, 1, 2]),
        JointProbability(Eq("page", True), Eq("escalate", True)),
    ],
)
```

Conceptual execution:

```text
state + proposition descriptions
               ↓
      unary/interaction utilities
               ↓
          ConstraintSpec
               ↓
 structured normalization or projection
               ↓
       canonical joint distribution
               ↓
 marginals / choices / scores / routes
```

The schema must distinguish:

- logical invariants from preferences;
- unconditional events from forced choices;
- exclusive groups from overlapping propositions;
- explicit `none` states from low confidence;
- hard constraints from soft costs.

## 11. Proposed model architecture

### 11.1 Shared encoder

Use the pretrained language model to encode:

- the state;
- proposition descriptions;
- variable descriptions;
- constraint descriptions when they contain semantic content.

### 11.2 Unary utility head

Produce evidence for each variable value or proposition:

\[
u_i=f_\theta(x,A_i).
\]

### 11.3 Optional interaction head

Produce pairwise or higher-order compatibility terms:

\[
v_{ij}=g_\theta(x,A_i,A_j).
\]

Known hard rules should not need to be relearned as interaction weights. Learned
interactions are for statistical dependencies not specified by the schema.

### 11.4 Constraint compiler

Compile the declarative schema into one of:

- a closed-form normalizer;
- a factor graph;
- a finite list of valid worlds;
- a Boolean/arithmetic circuit;
- a convex feasible set;
- an integer program for MAP inference.

### 11.5 Inference engine

Select exact or approximate inference based on graph structure and state-space
size. Return both the canonical representation and derived views, along with an
indicator of whether inference was exact.

## 12. Research hypotheses

### H1: Structural constraints remove logical violations

For declared invariants, hard-constrained heads should have zero violation rate
up to numerical precision, unlike independent per-question heads.

### H2: Coherence improves data efficiency

Equivalent views provide redundant supervision. Sharing one canonical
distribution may need fewer labeled examples than learning separate heads.

### H3: Constraints can improve accuracy

When invalid configurations otherwise receive nontrivial mass, redistributing
that mass among valid configurations may improve both record-level accuracy and
NLL.

### H4: Incorrect constraints cause systematic harm

Declaring independent events exclusive will force valid probability mass away
from multi-label worlds. Constraint correctness must therefore be treated as a
schema responsibility and tested explicitly.

### H5: Projection alone is insufficient for dependent events

Marginal projection will improve feasibility metrics but cannot match a model
trained to represent correlations and joint outcomes.

### H6: End-to-end constrained training calibrates better than post-hoc repair

Training through the constraint layer should outperform applying the same
constraints only after independently trained heads.

## 13. Evaluation plan

Evaluate every approach along separate axes.

### 13.1 Predictive quality

- per-field accuracy and macro-F1;
- exact-record accuracy;
- ranking metrics where appropriate;
- joint and marginal NLL;
- Brier score.

### 13.2 Calibration

- expected and adaptive calibration error;
- classwise reliability diagrams;
- marginal calibration;
- joint calibration for small structured spaces;
- calibration after conditioning or projection.

### 13.3 Constraint behavior

- hard violation rate;
- expected probability mass assigned to invalid worlds;
- distance between equivalent query views;
- simplex and hierarchy errors;
- correction magnitude for projected models.

### 13.4 Efficiency

- latency and throughput;
- memory consumption;
- exact versus approximate inference cost;
- scaling with variables, options, constraints, and graph treewidth.

### 13.5 Robustness

- option permutation;
- paraphrased variable and proposition descriptions;
- repeated equivalent questions;
- contradictory or unsatisfiable constraints;
- missing or incorrect relationship declarations;
- out-of-distribution schemas.

## 14. Candidate datasets and benchmarks

1. **TweetEval sentiment:** exclusive/exhaustive categorical and ordinal views;
   already implemented.
2. **Intent classification:** exclusive/exhaustive labels with optional
   out-of-scope state.
3. **Multi-label image or text tagging:** non-exclusive/non-exhaustive events.
4. **Incident routing:** hierarchical team, severity, paging, and escalation
   constraints.
5. **Workflow configuration:** cardinality, implication, and conditional option
   rules.
6. **Synthetic Boolean worlds:** controlled logical formulas with exact joint
   probabilities and known partition functions.
7. **Ordinal risk assessment:** monotone cumulative probabilities and
   asymmetric decision costs.

Synthetic data is especially useful for verifying exactness because the true
joint distribution and every valid marginal can be computed analytically.

## 15. Implementation roadmap

### Phase 1: relationship-aware closed forms

- Preserve the current exclusive/exhaustive shared softmax.
- Add exclusive/optional normalization with explicit `none`.
- Add independent paired binary logits.
- Add non-empty conditioned Bernoulli outputs.
- Define typed schemas and derived view APIs.
- Add algebraic unit and property tests for every relationship.

### Phase 2: finite structured worlds

- Enumerate valid worlds for small schemas.
- Compile exactly-one, at-most-one, at-least-one, implication, and cardinality
  constraints.
- Compute exact partition functions and marginals.
- Compare end-to-end constrained training with post-hoc projection.

### Phase 3: interaction models

- Add learned pairwise compatibility terms.
- Train on multi-field records.
- Measure when interactions improve over unary utilities plus hard rules.
- Analyze joint calibration.

### Phase 4: scalable inference

- Detect tractable tree/chain structure.
- Add variable elimination or belief propagation.
- Explore probabilistic circuits or weighted model counting for Boolean rules.
- Add approximate inference with error diagnostics.

### Phase 5: heterogeneous decision post-training

- Randomize schemas, option order, wording, and relationship types.
- Train across classification, multi-label, ordinal, and workflow tasks.
- Evaluate zero-shot generalization to unseen schemas.
- Compare directly with the original Clef head on quality, calibration,
  coherence, and latency.

## 16. Failure modes and open questions

- **Wrong declared semantics:** hard constraints make schema mistakes
  irreversible at inference time.
- **Unsatisfiable schemas:** the valid set may be empty; the API needs validation
  and explicit failure behavior.
- **Combinatorial explosion:** exact normalization may be infeasible for dense
  high-treewidth constraints.
- **Ambiguous forced choice:** choosing the “best” option may be a preference
  relation rather than conditioning on mutually exclusive events.
- **Unknown-world mass:** an explicit `none`, `other`, or abstain state is often
  necessary for calibrated open-world behavior.
- **Dependency misspecification:** independent Bernoulli outputs are coherent
  marginals but may be a poor joint model.
- **Calibration target:** marginal calibration, joint calibration, and utility
  calibration are not interchangeable.
- **Approximate inference:** returned probabilities should disclose approximation
  method and convergence/error information.
- **Constraint provenance:** schemas should record whether each rule is a logical
  invariant, policy requirement, learned dependency, or user preference.

## 17. Central design principle

The core proposal can be summarized as:

> Predict a calibrated distribution over valid underlying decisions first;
> derive binary, categorical, ordinal, and routing outputs afterward.

This is stronger than jointly encoding several questions and independently
normalizing their answers. It turns logical relationships from patterns the
network may learn into properties the probability model must satisfy.
