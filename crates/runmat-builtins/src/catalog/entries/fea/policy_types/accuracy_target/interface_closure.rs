use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTarget.interfaceClosure",
    summary: concat!(
        "Target closure of coupled interfaces (transferred flux and traction balance) for ",
        "accuracy acceptance.",
    ),
    description: concat!(
        "`fea.AccuracyTarget.interfaceClosure()` returns the `fea.AccuracyTarget` whose ",
        "`value` is `interface_closure`. It assesses how well quantities agree across a ",
        "coupled or projected interface.",
    ),
    keywords: &["fea", "accuracy", "target", "interface closure", "acceptance"],
    related: &["fea.AccuracyTarget.observation", "fea.AccuracyTarget.quantity", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "An interface-closure criterion accepts only coupling, interface-projection, or ",
                    "field-transfer error contributions; other contributions fail study validation.",
                ),
                super::TARGET_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Use this target in coupled steps, such as conjugate heat transfer or fluid-structure",
                    " interaction, when the acceptance measure is the mismatch of exchanged quantities at",
                    " the interface.",
                ),
                super::TARGET_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "interface-closure",
            "Select the interface closure target",
            "target = fea.AccuracyTarget.interfaceClosure();",
            concat!(
                "assert(strcmp(class(target), 'fea.AccuracyTarget'));\n",
                "assert(target.value == \"interface_closure\");",
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
    FEA_ACCURACY_TARGET_INTERFACE_CLOSURE_CATALOG_ENTRY,
    name: "fea.AccuracyTarget.interfaceClosure",
    result: classes::ACCURACY_TARGET,
    signature: "target = fea.AccuracyTarget.interfaceClosure()",
    inputs: [
    ],
    output: ("target", "Typed fea.AccuracyTarget value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
