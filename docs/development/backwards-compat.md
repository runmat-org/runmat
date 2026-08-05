---
title: "Semantic Compatibility Engineering Policy"
category: "Development"
section: "14.7"
last_updated: "August 4, 2026"
---

# Semantic Compatibility Engineering Policy

RunMat is an independent runtime for MATLAB-language source code. Compatibility work is based on publicly available documentation and independent engineering. This policy explains how the project selects a documented semantic reference, evaluates behavior that has changed across releases, and preserves older source code where practical.

MATLAB is a registered trademark of The MathWorks, Inc. RunMat is not affiliated with or endorsed by The MathWorks, Inc.

## Scope

This policy applies to observable behavior in RunMat-supported surfaces, including:

- language syntax and evaluation semantics;
- values, classes, shapes, indexing, and assignment;
- builtin inputs, outputs, warnings, and errors;
- persistence and documented file-format behavior; and
- CPU and accelerated execution of supported operations.

This policy does not:

- claim complete coverage of the MATLAB language or product family;
- make implementation details of another runtime part of RunMat's contract;
- reproduce product licensing or packaging;
- require RunMat to share another product's performance characteristics, resource limits, or internal architecture; or
- turn undocumented behavior into a compatibility requirement.

Feature support is documented separately. A compatibility target identifies the reference used for matching semantic behavior. It does not imply that every toolbox or feature in that release is implemented, but language behavior will aim to match compatibility with existing code written for that release.

## Current semantic target

The current semantic compatibility target for RunMat is **MATLAB R2026a**.

## Compatibility vocabulary

**Semantic target**: The pinned release whose observable or online documented behavior is the primary reference for supported behavior.

**Legacy compatibility envelope**: Documented behavior from earlier releases that RunMat can preserve without conflicting with the semantic target.

**Conflict**: A case where the same supported program and inputs cannot exhibit both the target behavior and an older behavior.

**RunMat extension**: Deliberate behavior or syntax provided by RunMat that is outside the documented target surface.

**Observable behavior**: A program-visible result such as a value, class, shape, side effect, warning, error, or accepted call form. Internal algorithms and data structures are not observable behavior for this policy.

## Decision order

When implementing or reviewing a supported surface, use this order:

1. Follow behavior explicitly documented for the semantic target.
2. Follow documented target-release errors, warnings, and unsupported cases when they are material to program behavior.
3. Preserve older call forms and behaviors that the target still accepts.
4. Preserve additional documented legacy behavior when it does not conflict with the target.
5. Treat behavior beyond that surface as a RunMat extension only after an explicit engineering decision.
6. When reliable public evidence is insufficient, record the ambiguity and use a safe, deterministic provisional behavior rather than claiming parity.

Current target behavior takes precedence in a genuine conflict. This rule keeps the runtime forward-facing without unnecessarily excluding older source code.

## Backward compatibility policy

RunMat aims to execute existing MATLAB-language source code across a broad compatibility envelope. That aim is applied behavior by behavior rather than by declaring that every script from a particular historical release is supported.

The RunMat project should:

- continue to support an older call form when the semantic target still supports it;
- distinguish "not recommended" from removed or unsupported functionality;
- avoid requiring source rewrites solely to use a newer preferred spelling;
- preserve nonconflicting legacy syntax and semantics when the implementation cost and maintenance burden are reasonable;
- preserve documented interchange formats where the relevant RunMat feature supports them; and
- test legacy behavior that RunMat intentionally retains.

The project is not required to preserve an older behavior when it conflicts with the semantic target. In such a case, the target behavior applies unless the project explicitly adopts a mode-gated RunMat extension.

There is no blanket earliest-supported MATLAB release. A historical behavior is inside the compatibility envelope only when it is supported by adequate public evidence, relevant to a RunMat surface, and not displaced by this policy.

## Compatibility modes

RunMat's `runmat`, `matlab`, and `strict` modes are language-policy modes, not intended to emulate different MATLAB releases.

- Supported MATLAB-language programs should have consistent numeric values, classes, shapes, and indexing results across modes.
- `matlab` mode excludes RunMat-only language features where accepting them would contradict the mode's purpose and controls MATLAB-style error identifiers where implemented.
- `strict` mode controls permissive syntax, including command-style calls. It does not select historical numeric semantics.
- `runmat` mode may expose deliberate extensions, subject to the extension rules below.

Release-specific runtime modes should not be introduced without a separate design review. They would multiply the semantic and testing surface and are not part of the current compatibility strategy.

## RunMat extensions

RunMat extends the MATLAB language with intentional language features that are not part of MathWork's MATLAB. Language extensions are designed to be a superset of MATLAB semantics, and are designed to not conflict with existing MATLAB code. Language extensions can be disabled with the compatability configuration flag described in the [Configuration Reference](/docs/runtime/getting-started/config). An extension must be classified before it becomes part of supported behavior.

| Classification | Meaning | Policy |
| --- | --- | --- |
| Safe extension | Adds capability without changing a program accepted by the semantic target | May be available in `runmat` mode and documented as an extension |
| Mode-gated extension | Accepting it conflicts with documented target behavior | May be considered for `runmat` mode; must not silently alter `matlab` mode |
| Incompatible divergence | Changes the result of a program accepted by the target | Do not introduce without a separate compatibility decision |

The existence of a useful internal representation or implementation path does not by itself justify exposing a new language behavior. Extensions must be intentional, documented, and tested.

Each mode-gated builtin signature or option must register a stable extension identifier, required compatibility mode, description, and error identifier in builtin metadata, and its implementation must enforce that same declaration through the shared runtime compatibility guard. This keeps documentation, tooling, and execution policy queryable from one registration record while allowing argument-dependent extension forms within an otherwise MATLAB-compatible builtin.

Typed sparse integer storage is one mode-gated extension. MATLAB-compatible modes accept only documented `double`, `single`, and `logical` sparse payloads and reject construction, conversion, restoration, indexed access or assignment, and operation results that would expose sparse integer storage; `runmat` mode may preserve exact signed or unsigned sparse integer payloads. Sparse values remain host-resident in every mode because the current acceleration interface exposes dense tensor handles rather than native sparse device storage.

The legacy NaN-aware Statistics and Machine Learning Toolbox reductions (`nanmean`, `nansum`, `nanmin`, `nanmax`, `nanmedian`, `nanstd`, and `nanvar`) are not unconditional aliases for their modern replacements because their documented call forms and accepted classes differ. MATLAB-compatible modes retain the documented floating-input surface; typed-integer data or controls accepted through modern RunMat reduction machinery are individually registered and mode-gated extensions, while typed-integer data remains rejected for `nanstd` and `nanvar`. Compatibility checks inspect resident integer handle metadata before provider dispatch and do not gather device data merely to classify the call.

Canonical reductions retain their documented per-form integer contracts rather than applying a blanket numeric rule: `sum` and `prod` return double by default and preserve class with saturating `"native"` arithmetic; `median`, `min`, `max`, and both `bounds` outputs preserve integer class; and `std` and `var` reject integer data in every mode. The documented `std` and `var` data and weight domains are floating, so RunMat's useful typed-integer normalization and dimension controls are registered mode-gated extensions rather than silently widening the MATLAB-compatible surface.

The cumulative reducers preserve integer class and shape: `cumsum` and `cumprod` apply per-step saturating native arithmetic, while `cummin` and `cummax` select exact values and expose one public output in the current semantic target. Interactive GPU input is documented for all four names, but explicit missing-value flags are not supported on GPU for `cumprod`, `cummin`, or `cummax`; RunMat's supported GPU flag forms are therefore registered mode-gated extensions.

The descriptive and order-statistic builtins retain their documented per-form numeric domains. `mode` accepts every integer class and preserves the class of its modal values and ties while returning double frequencies; `movmad` accepts integer observations and controls on its currently supported scalar-window surface and computes double deviations. `mad`, `range`, `quantile`, and `prctile` document floating or logical data domains rather than typed-integer observations, so RunMat's exact integer ordering or extrema followed by double deviation, interpolation, or range output is mode-gated. Typed-integer controls outside each documented form, explicit `range` missing-value flags, GPU `range` all-or-vector-dimension forms, and GPU `movmad` windows longer than 31 are separately registered extensions, allowing each restriction to be enforced before gathering resident data.

The adjacent descriptive functions likewise retain their individually documented domains: `geomean`, `harmmean`, `skewness`, `kurtosis`, `tiedrank`, and numeric `tabulate` document single/double data; `rms` additionally documents logical and character data; and `rmse` documents single/double forecast, actual, and weight arrays. RunMat's typed real/complex integer computation forms are independently mode-gated before materializing into double mean, moment, residual, magnitude, rank, or weight domains. Typed controls outside the documented forms, GPU all-or-vector-dimension skewness/kurtosis, and GPU `tabulate` are separate gates. Integer `tabulate` is intentionally exact and returns a heterogeneous cell table with same-class integer values plus double counts and percentages; it expands absent positive levels only for a bounded practical range and otherwise groups the exact observed values.

`corr` documents single/double observation matrices and single/double nonnegative observation weights. RunMat's typed-integer observation and weight forms are independent mode-gated extensions: exact same-class ordering is retained for Spearman and Kendall correlation, Pearson centers integer columns through exact differences before entering the double correlation domain, and typed weights are validated exactly before floating weighted computation. Weighted calls return `NaN` p-values on the same surface as the documented form, and supported resident inputs use explicit gather fallback rather than changing the compatibility classification.

`cov` documents single/double observation data, a scalar single/double normalization weight of zero or one, and missing-row controls. RunMat independently mode-gates typed-integer data, logical data, typed-integer/logical normalization, and its broader vector-of-observation-weights form before provider dispatch or gather. Integer variables are centered through exact differences before entering double covariance arithmetic, supported resident integer data uses explicit gather fallback, and vector weights remain a RunMat extension for every element class rather than being confused with the documented scalar normalization weight.

`corrcoef` documents real or complex single/double observation data, equal-size paired inputs that are vectorized into two variables, `Rows` and floating `Alpha` name-value controls, and one, two, or four public outputs. RunMat independently mode-gates typed-integer and logical observation data before provider dispatch or gather, centers real and complex integer components through exact differences before floating correlation, rejects typed-integer `Alpha` because no integer satisfies its documented open interval, preserves complex one-output correlation, and keeps supported real single-output GPU calls resident with the same vector and paired geometry.

## Evidence policy

Compatibility decisions should be based on publicly available information and literature only. RunMat is a clean-room implementation of MATLAB compatable syntax, and as a result we do not rely on proprietary binaries, disassembly, or reverse engineering. Prefer evidence in this order:

1. Publically available explicit behavior documentation.
2. Publicly available release notes and documented version history.
3. Publicly available bug reports.
4. Publicly available Answers and authored explanatory material.
5. Independent technical sources that corroborate or clarify primary sources.
6. Community discussion as a signal about adoption or risk, not as sole authority for a semantic rule.

## Recording a semantic decision

A nontrivial decision should record enough information to reproduce and validate semantic behavior with tests, with notes within comments relevant:

| Field | Purpose |
| --- | --- |
| Target release | Release used as the primary reference |
| Surface | Operator, builtin, syntax, storage, or execution environment |
| Target behavior | Documented behavior for the target |
| Legacy behavior | Relevant earlier behavior, if different |
| Change release | Release in which public documentation reports a change |
| Conflict | Whether target and legacy behavior can coexist |
| Applicability | Core language, toolbox, accelerator, code generation, or other surface |
| Evidence | Direct public sources and whether the conclusion is explicit or inferred |
| RunMat policy | Behavior selected for RunMat |
| Tests | Conformance evidence in the repository |
| Remaining ambiguity | Unresolved or weakly supported details |

## Related documentation

- [Language Compatibility](/docs/runtime/getting-started/compatability)
- [Configuration Reference](/docs/runtime/getting-started/config)
- [Testing Strategy](/docs/runtime/development/testing)
