use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RecoveryPolicy.volumeWeightedAverage",
    summary: "Average integration-point, element, or cell values weighted by volume.",
    description: concat!(
        "`fea.RecoveryPolicy.volumeWeightedAverage()` returns the `fea.RecoveryPolicy` whose ",
        "`value` is `volume_weighted_average`. Volume-weighted averaging combines ",
        "contributions from integration points, elements, or cells by their volumes, the ",
        "usual way to obtain smooth nodal stress.",
    ),
    keywords: &["fea", "observation", "recovery policy", "volume weighted average"],
    related: &["fea.RecoveryPolicy.native", "fea.RecoveryPolicy.extrapolate", "fea.RecoveryPolicy.areaWeightedAverage", "fea.RecoveryPolicy.conservativeProjection", "fea.FieldLocation.node", "fea.FieldLocation.integrationPoint"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Volume-weighted averaging combines contributions from integration points, elements, ",
                    "or cells by their volumes, the usual way to obtain smooth nodal stress.",
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
            "volume-weighted-average",
            "Average integration-point, element, or cell values weighted by volume",
            "recovery = fea.RecoveryPolicy.volumeWeightedAverage();",
            concat!(
                "assert(strcmp(class(recovery), 'fea.RecoveryPolicy'));\n",
                "assert(recovery.value == \"volume_weighted_average\");",
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
    FEA_RECOVERY_POLICY_VOLUME_WEIGHTED_AVERAGE_CATALOG_ENTRY,
    name: "fea.RecoveryPolicy.volumeWeightedAverage",
    result: classes::RECOVERY_POLICY,
    signature: "recovery = fea.RecoveryPolicy.volumeWeightedAverage()",
    inputs: [
    ],
    output: ("recovery", "Typed fea.RecoveryPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
