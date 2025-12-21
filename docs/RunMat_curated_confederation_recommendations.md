RunMat Curated Confederation Recommendations
=============================================================

Quick background (why Julia and ModelingToolkit matter)
-------------------------------------------------------
Julia's ModelingToolkit is a system where:
- The core math engine is public and well-documented.
- Other people can add new math transformations and solvers without editing the core.
- The core documentation can list and recognize those extensions.

RunMat wants similar benefits, but with a stronger emphasis on MATLAB compatibility.

Sources referenced:
- `docs/RunMat_curated_confederation_model.md`
- `docs/SYMBOLIC_ENGINE_ASSESSMENT.md`
- `docs/SYMBOLIC_SESSION_STATUS.md`
- `Symbolic_Engine_Plan.txt`

Matches (what RunMat already has)
---------------------------------
1) Extensible transformation pipeline
RunMat already processes symbolic expressions in clear stages, which
  makes it possible to add new stages later.
- Evidence: The staged normalization pipeline is explicit and composable.

2) Shared symbolic representation
There is a common, public way to represent symbolic math in the system.
- Evidence: `Value::Symbolic` is first-class and integrated, with a clear compilation path.

3) Transparent transformations
It is possible to see what transformations are applied to an expression.
- Evidence: "Proof-carrying simplification" and pass-based normalization are documented.

4) MATLAB compatibility is non-negotiable
RunMat must stay MATLAB-compatible by default.
- Evidence: Both assessment and session status emphasize MATLAB compatibility.

Gaps (what is missing or unclear)
---------------------------------
1) No clear boundary between "core" and "extensions"
There is no documented line between what is always included and
  what is optional or experimental.

2) No transformation registry
There is no catalog that explains the available transformation steps,
  what each one does, and when it is safe to use.
- Why it matters: Without this, external contributors cannot reliably plug in or be listed.

3) No guide for external extensions
There is no documented process for third parties to ship extensions as
  separate crates and have RunMat recognize them.

4) No published stability guarantees for the symbolic IR
It is not clear what parts of the symbolic representation are stable
  across versions, which discourages external extensions.

5) No formal correctness and safety declarations
Transformations do not declare whether they are always correct,
  only correct under assumptions, or just heuristics.

6) No curated extension list
There is no official list of supported or recommended extensions.

Recommendations (ordered)
-------------------------
1) Define a two-tier ecosystem (core vs extension)
- Core: MATLAB-compatible, stable, always-on behavior.
- Extension: optional, experimental, or research-grade features.

2) Add a transformation registry
- Create a catalog of transformation passes with metadata:
  name, purpose, assumptions, correctness level, performance impact, and ordering rules.
- Use this registry to build the default pipeline and to list extensions.

3) Publish an extension packaging and discovery guide
- Describe how to create an external crate, how RunMat loads it, and how users opt in.

4) Publish a symbolic IR stability policy
- State which APIs are stable and how long they remain stable.
- Provide a deprecation policy for breaking changes.

5) Formalize correctness guardrails
- Require each transformation to declare whether it is:
  always correct, assumption-dependent, or heuristic.

6) Curate extension documentation
- Maintain a list of extensions with clear labels:
  core-compatible, research-grade, experimental.

Automatic Differentiation and Related Transforms (Compact)
----------------------------------------------------------
Short answer: most of the ideas in `docs/Automatic_Diff_ETC.md` are possible in RunMat,
but not all are equally near-term. RunMat already has a symbolic IR, a staged
normalization pipeline, and a bytecode compiler, which are strong foundations for
transformation-based features.

Near-term candidates:
- Symbolic forward-mode AD on expression trees.
- Jacobian sparsity detection via dependency ("can influence") sets.
- Interval-based uncertainty propagation.

Longer-term candidates:
- Reverse-mode AD at IR/bytecode level for performance.
- DAE index reduction as a research-grade extension.
- Auto-parallelization as an opt-in optimization pass.

Reference: `docs/Automatic_Diff_ETC.md` for the full background and rationale.

Single-Document Implementation Details
--------------------------------------
To avoid scattering information, the following sections consolidate the policy and guide
content in one place.

Extension Tier Policy (Core vs Extension)
-----------------------------------------
Core tier (MATLAB-compatible):
- Stable, conservative behavior matching MATLAB semantics.
- Included by default; used in compatibility tests.
- Changes require explicit deprecation and compatibility review.

Extension tier (opt-in):
- Experimental, research-grade, or high-risk transforms and solvers.
- Disabled by default; explicit opt-in required.
- Versioned independently and can evolve faster.

Classification labels:
- core-compatible (safe, stable, MATLAB-aligned)
- research-grade (validated but may require assumptions)
- experimental (no stability guarantee)

Transformation Registry
-----------------------
Purpose: Make transformations discoverable, composable, and safe to integrate.

Registry entry fields (minimum):
- name: stable identifier
- purpose: what it does
- preconditions: required assumptions (e.g., nonzero denominators)
- soundness: semantics-preserving | assumption-dependent | heuristic
- numeric safety: safe | risky | unknown
- performance impact: low | medium | high
- ordering: before/after constraints
- tier: core-compatible | research-grade | experimental

Default pipeline behavior:
- Constructed from registry entries.
- Filters by tier and policy (core-only by default).
- Includes provenance in proof-carrying output where available.

Symbolic IR Stability Policy
----------------------------
Scope:
- Symbolic expression representation, normalization pass APIs, bytecode compiler hooks.

Versioning rules (proposal):
- IR and pass APIs are stable within a minor version.
- Breaking changes require a major version bump and deprecation cycle.

Deprecation process:
- Announce deprecation with alternatives.
- Provide migration guidance and a full release cycle before removal.

Extensions Guide (Packaging and Discovery)
------------------------------------------
Packaging:
- External passes and solvers live in separate crates.
- Naming convention: `runmat-ext-<feature>` or `runmat-pass-<name>`.
- Each extension declares tier and compatibility in its manifest.

Discovery:
- Opt-in via config file or environment variable.
- Registry loads only whitelisted extensions to protect MATLAB defaults.

Documentation:
- Maintain a curated list with tier labels and brief capability summaries.
- Explicitly note assumptions and any non-MATLAB behavior.
