use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.NumericalEquivalencePolicy.deterministic",
    summary: "Require executions of a step to produce identical numerical results.",
    description: concat!(
        "`fea.NumericalEquivalencePolicy.deterministic()` returns the policy whose `value` is",
        " `deterministic`: executions of the step must reproduce the same numbers, with no ",
        "tolerance for variation.",
    ),
    keywords: &["fea", "accuracy", "equivalence", "deterministic", "reproducibility"],
    related: &["fea.NumericalEquivalencePolicy.relativeTolerance", "fea.NumericalEquivalencePolicy.absoluteTolerance", "fea.AccuracyPlan.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The policy takes no arguments and always passes plan validation. Choose it when ",
                    "results must be bit-reproducible, and only for formulations and placements whose ",
                    "reductions are deterministic.",
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
            "deterministic",
            "Require identical results across executions",
            "policy = fea.NumericalEquivalencePolicy.deterministic();",
            concat!(
                "assert(strcmp(class(policy), 'fea.NumericalEquivalencePolicy'));\n",
                "assert(policy.value == \"deterministic\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Does deterministic make a parallel run slower?",
            concat!(
                "The policy does not change how work is scheduled; it changes what the comparison ",
                "accepts. Use a tolerance policy when partitioned reductions may legitimately differ ",
                "in the last bits.",
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
    FEA_NUMERICAL_EQUIVALENCE_POLICY_DETERMINISTIC_CATALOG_ENTRY,
    name: "fea.NumericalEquivalencePolicy.deterministic",
    result: classes::NUMERICAL_EQUIVALENCE_POLICY,
    signature: "policy = fea.NumericalEquivalencePolicy.deterministic()",
    inputs: [
    ],
    output: ("policy", "Typed fea.NumericalEquivalencePolicy."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
