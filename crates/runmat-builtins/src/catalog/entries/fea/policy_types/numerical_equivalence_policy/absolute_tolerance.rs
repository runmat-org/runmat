use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.NumericalEquivalencePolicy.absoluteTolerance",
    summary: "Allow a bounded absolute difference between executions of a step.",
    description: concat!(
        "`fea.NumericalEquivalencePolicy.absoluteTolerance(value)` returns a policy whose ",
        "stored value is the struct `absolute_tolerance: value`. Two executions agree when ",
        "their compared values differ by at most `value`, in the canonical SI units of the ",
        "compared values.",
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
            "absolute-tolerance",
            "Allow bounded variation between executions",
            "policy = fea.NumericalEquivalencePolicy.absoluteTolerance(1e-12);",
            concat!(
                "assert(strcmp(class(policy), 'fea.NumericalEquivalencePolicy'));\n",
                "assert(policy.value.absolute_tolerance == 1e-12);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Which unit does the absolute bound use?",
            concat!(
                "The canonical SI unit of each compared value, the same units RunMat stores results ",
                "in. Prefer a relative bound when one plan covers quantities with different units.",
            ),
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
    FEA_NUMERICAL_EQUIVALENCE_POLICY_ABSOLUTE_TOLERANCE_CATALOG_ENTRY,
    name: "fea.NumericalEquivalencePolicy.absoluteTolerance",
    result: classes::NUMERICAL_EQUIVALENCE_POLICY,
    signature: "policy = fea.NumericalEquivalencePolicy.absoluteTolerance(value)",
    inputs: [
        (
            "value",
            NumericScalar,
            concat!(
                "Finite bound in the canonical SI unit of the compared values; must be positive when ",
                "the study is validated.",
            ),
            PolicyArg::RealScalar,
        ),
    ],
    output: ("policy", "Typed fea.NumericalEquivalencePolicy."),
    integer_capabilities: &integer::TOLERANCE_VALUE_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
