use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RecoveryPolicy.native",
    summary: "Report values exactly where the solver computes them.",
    description: concat!(
        "`fea.RecoveryPolicy.native()` returns the `fea.RecoveryPolicy` whose `value` is ",
        "`native`. Native recovery performs no transfer.",
    ),
    keywords: &["fea", "observation", "recovery policy", "native"],
    related: &["fea.RecoveryPolicy.extrapolate", "fea.RecoveryPolicy.volumeWeightedAverage", "fea.RecoveryPolicy.areaWeightedAverage", "fea.RecoveryPolicy.conservativeProjection", "fea.FieldLocation.node", "fea.FieldLocation.integrationPoint"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Native recovery performs no transfer. It is required when the source and output ",
                    "locations are the same, and allowed for a global output produced by a spatial ",
                    "reduction.",
                ),
                super::ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "native",
            "Report values exactly where the solver computes them",
            "recovery = fea.RecoveryPolicy.native();",
            concat!(
                "assert(strcmp(class(recovery), 'fea.RecoveryPolicy'));\n",
                "assert(recovery.value == \"native\");",
            ),
        ),
    ],
    faqs: &[
        super::FAQ,
    ],
    links: &[
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
        page("Outputs, fields, and observations", "/docs/simulation/language/outputs-fields-and-observations"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/recovery_policy/tests.rs::recovery_policy_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_RECOVERY_POLICY_NATIVE_CATALOG_ENTRY,
    name: "fea.RecoveryPolicy.native",
    result: classes::RECOVERY_POLICY,
    signature: "recovery = fea.RecoveryPolicy.native()",
    inputs: [
    ],
    output: ("recovery", "Typed fea.RecoveryPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
