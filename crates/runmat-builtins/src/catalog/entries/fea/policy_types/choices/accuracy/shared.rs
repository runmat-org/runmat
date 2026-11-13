//! Documentation facts shared by every identity of this policy family.

use crate::fea_contract::policy_types::documentation::{page, section, source, unit_test};
use crate::*;

pub(super) const USE_CRITERION: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Error contributions and required evidence are arguments of `fea.AccuracyCriterion.new(id, target, contribution, tolerance, evidence, mandatory)`. Criteria are collected by `fea.AccuracyPlan.new(id, criteria, unavailableEvidence, exceededLimit, equivalence, resources)`, and the plan is attached to a step with `fea.ProcedureStep.configure(..., accuracy, restart)`.",
        "After the solve, RunMat matches each criterion to evidence with the same target, contribution, and evidence kind. Matching evidence within tolerance satisfies the criterion; evidence beyond tolerance does not; no matching evidence leaves it indeterminate. Mandatory criteria decide the plan's status.",
    ],
);
pub(super) const USE_PLAN: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the policy to `fea.AccuracyPlan.new(id, criteria, unavailableEvidence, exceededLimit, equivalence, resources)` and attach the plan to a step with `fea.ProcedureStep.configure(..., accuracy, restart)`.",
        "A satisfied plan always allows publication. An indeterminate result is published as degraded only with `markIndeterminate`; an unsatisfied result is always blocked.",
    ],
);
pub(super) const EVIDENCE_PRODUCERS: BuiltinDocumentationSection = section(
    "Evidence RunMat produces today",
    &[
        "Global-equilibrium algebraic residual bounds and conservation balances come from linear structural, thermal, electrical, magnetostatic, Maxwell harmonic, acoustic harmonic, and steady fluid solves; nonlinear solid and thermal solves report a nonlinear residual bound. Modal response-spectrum and random-response steps report modal-truncation and frequency-sampling convergence studies on the numerical result. Structural adaptive refinement reports spatial-discretization and recovery convergence, an a-posteriori recovery estimator, and field-transfer bounds.",
        "A criterion with no producer is reported indeterminate rather than silently passed; the plan's `fea.UnavailableEvidencePolicy` then decides what happens.",
    ],
);
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[
    page(
        "Procedure sequences, observations, and accuracy",
        "/docs/simulation/study-contracts",
    ),
    page("Trust and evidence", "/docs/simulation/trust"),
];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    source(
        "Accuracy policy constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study_policies/accuracy.rs",
    ),
    source(
        "Accuracy plan validation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/core/src/validate_accuracy.rs",
    ),
    source(
        "Accuracy evaluation and publication gate",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/fea/src/results/accuracy/mod.rs",
    ),
];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[
    unit_test(
        "Accuracy choices return typed catalog values",
        "crates/runmat-runtime/src/builtins/fea/study_policies/accuracy/tests.rs::accuracy_choices_return_typed_catalog_values",
    ),
    unit_test(
        "Catalog contract conformance",
        "crates/runmat-builtins/src/catalog/entries/fea/policy_types/choices/accuracy/tests.rs::accuracy_policy_choices_conform",
    ),
];
