use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTarget.executionEquivalence",
    summary: "Target agreement between executions of the same step for accuracy acceptance.",
    description: concat!(
        "`fea.AccuracyTarget.executionEquivalence()` returns the `fea.AccuracyTarget` whose ",
        "`value` is `execution_equivalence`. It assesses whether serial, partitioned, ",
        "retried, or differently placed executions agree within the plan's ",
        "numerical-equivalence policy.",
    ),
    keywords: &["fea", "accuracy", "target", "execution equivalence", "acceptance"],
    related: &["fea.AccuracyTarget.observation", "fea.AccuracyTarget.quantity", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "An execution-equivalence criterion accepts only the execution-equivalence error ",
                    "contribution, and its evidence must be a reference comparison or residual bound. The",
                    " tolerated variation is governed by the plan's `fea.NumericalEquivalencePolicy`.",
                ),
                super::TARGET_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Use this target when a result must be reproducible across placements, for example ",
                    "when the same step runs locally and on a cluster.",
                ),
                super::TARGET_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "execution-equivalence",
            "Select the execution equivalence target",
            "target = fea.AccuracyTarget.executionEquivalence();",
            concat!(
                "assert(strcmp(class(target), 'fea.AccuracyTarget'));\n",
                "assert(target.value == \"execution_equivalence\");",
            ),
        ),
    ],
    faqs: &[
        super::TARGET_FAQ,
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_target/tests.rs::accuracy_target_identities_conform",
        ),
        unit_test(
            "Targets preserve typed values and reject untyped inputs",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::targets_preserve_typed_values_and_reject_empty_or_untyped_inputs",
        ),
        unit_test(
            "Accuracy plan validation rejects nonpositive tolerances",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_nonpositive_tolerance",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_TARGET_EXECUTION_EQUIVALENCE_CATALOG_ENTRY,
    name: "fea.AccuracyTarget.executionEquivalence",
    result: classes::ACCURACY_TARGET,
    signature: "target = fea.AccuracyTarget.executionEquivalence()",
    inputs: [
    ],
    output: ("target", "Typed fea.AccuracyTarget value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
