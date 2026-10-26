use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.NumericalEquivalencePolicy.relativeTolerance",
    summary: "Allow a bounded relative difference between executions of a step.",
    description: concat!(
        "`fea.NumericalEquivalencePolicy.relativeTolerance(value)` returns a policy whose ",
        "stored value is the struct `relative_tolerance: value`. Two executions agree when ",
        "their results differ by at most `value` relative to the result magnitude.",
    ),
    keywords: &["fea", "accuracy", "equivalence", "reproducibility", "tolerance"],
    related: &["fea.NumericalEquivalencePolicy.deterministic", "fea.AccuracyTarget.executionEquivalence", "fea.AccuracyPlan.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`value` must be a finite real scalar; double and all built-in integer classes are ",
                    "accepted. The constructor stores it as binary64. When the study is validated the ",
                    "value must also be positive; zero or a negative value fails with an invalid accuracy",
                    " plan error.",
                ),
                super::EQUIVALENCE_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::EQUIVALENCE_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "relative-tolerance",
            "Allow bounded variation between executions",
            "policy = fea.NumericalEquivalencePolicy.relativeTolerance(1e-10);",
            concat!(
                "assert(strcmp(class(policy), 'fea.NumericalEquivalencePolicy'));\n",
                "assert(policy.value.relative_tolerance == 1e-10);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "What does the baseline plan use?",
            "`fea.AccuracyPlan.baseline` uses `relativeTolerance(1e-10)`.",
        ),
    ],
    links: &[
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/accuracy.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/numerical_equivalence_policy/tests.rs::numerical_equivalence_identities_conform",
        ),
        unit_test(
            "Equivalence policies store finite scalars",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::tolerances_and_equivalence_store_finite_scalars_and_reject_non_numeric_inputs",
        ),
        unit_test(
            "Accuracy plan validation rejects nonpositive tolerances",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_nonpositive_tolerance",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_NUMERICAL_EQUIVALENCE_POLICY_RELATIVE_TOLERANCE_CATALOG_ENTRY,
    name: "fea.NumericalEquivalencePolicy.relativeTolerance",
    result: classes::NUMERICAL_EQUIVALENCE_POLICY,
    signature: "policy = fea.NumericalEquivalencePolicy.relativeTolerance(value)",
    inputs: [
        (
            "value",
            NumericScalar,
            concat!(
                "Finite dimensionless bound, for example 1e-10; must be positive when the study is ",
                "validated.",
            ),
            PolicyArg::RealScalar,
        ),
    ],
    output: ("policy", "Typed fea.NumericalEquivalencePolicy."),
    integer_capabilities: &integer::TOLERANCE_VALUE_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
