use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTarget.globalEquilibrium",
    summary: concat!(
        "Target global equilibrium (the algebraic residual) of a step for accuracy ",
        "acceptance.",
    ),
    description: concat!(
        "`fea.AccuracyTarget.globalEquilibrium()` returns the `fea.AccuracyTarget` whose ",
        "`value` is `global_equilibrium`. It assesses how well the discrete system is ",
        "balanced.",
    ),
    keywords: &["fea", "accuracy", "target", "global equilibrium", "acceptance"],
    related: &["fea.AccuracyTarget.observation", "fea.AccuracyTarget.quantity", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "A global-equilibrium criterion accepts only algebraic, nonlinear, or coupling error ",
                    "contributions; other contributions fail study validation. The default ",
                    "`fea.AccuracyPlan.baseline` uses this target with a relative residual ",
                    "bound of 1e-8.",
                ),
                super::TARGET_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Use this target to require a residual or conservation-balance bound on the solved ",
                    "system, separately from discretization error in any reported quantity.",
                ),
                super::TARGET_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "global-equilibrium",
            "Select the global equilibrium target",
            "target = fea.AccuracyTarget.globalEquilibrium();",
            concat!(
                "assert(strcmp(class(target), 'fea.AccuracyTarget'));\n",
                "assert(target.value == \"global_equilibrium\");",
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
    FEA_ACCURACY_TARGET_GLOBAL_EQUILIBRIUM_CATALOG_ENTRY,
    name: "fea.AccuracyTarget.globalEquilibrium",
    result: classes::ACCURACY_TARGET,
    signature: "target = fea.AccuracyTarget.globalEquilibrium()",
    inputs: [
    ],
    output: ("target", "Typed fea.AccuracyTarget value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
