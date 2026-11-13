use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RecoveryPolicy.conservativeProjection",
    summary: "Transfer values with a conservative projection.",
    description: concat!(
        "`fea.RecoveryPolicy.conservativeProjection()` returns the `fea.RecoveryPolicy` whose",
        " `value` is `conservative_projection`. Conservative projection transfers values ",
        "between any pair of locations while preserving the integral of the quantity.",
    ),
    keywords: &["fea", "observation", "recovery policy", "conservative projection"],
    related: &["fea.RecoveryPolicy.native", "fea.RecoveryPolicy.extrapolate", "fea.RecoveryPolicy.volumeWeightedAverage", "fea.RecoveryPolicy.areaWeightedAverage", "fea.FieldLocation.node", "fea.FieldLocation.integrationPoint"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Conservative projection transfers values between any pair of locations while ",
                    "preserving the integral of the quantity.",
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
            "conservative-projection",
            "Transfer values with a conservative projection",
            "recovery = fea.RecoveryPolicy.conservativeProjection();",
            concat!(
                "assert(strcmp(class(recovery), 'fea.RecoveryPolicy'));\n",
                "assert(recovery.value == \"conservative_projection\");",
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
    FEA_RECOVERY_POLICY_CONSERVATIVE_PROJECTION_CATALOG_ENTRY,
    name: "fea.RecoveryPolicy.conservativeProjection",
    result: classes::RECOVERY_POLICY,
    signature: "recovery = fea.RecoveryPolicy.conservativeProjection()",
    inputs: [
    ],
    output: ("recovery", "Typed fea.RecoveryPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
