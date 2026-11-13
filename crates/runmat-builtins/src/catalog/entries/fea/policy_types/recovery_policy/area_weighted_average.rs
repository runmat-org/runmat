use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RecoveryPolicy.areaWeightedAverage",
    summary: "Average face or interface values weighted by area.",
    description: concat!(
        "`fea.RecoveryPolicy.areaWeightedAverage()` returns the `fea.RecoveryPolicy` whose ",
        "`value` is `area_weighted_average`. Area-weighted averaging combines face or ",
        "interface values by their areas; it requires a face or interface source location.",
    ),
    keywords: &["fea", "observation", "recovery policy", "area weighted average"],
    related: &["fea.RecoveryPolicy.native", "fea.RecoveryPolicy.extrapolate", "fea.RecoveryPolicy.volumeWeightedAverage", "fea.RecoveryPolicy.conservativeProjection", "fea.FieldLocation.node", "fea.FieldLocation.integrationPoint"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Area-weighted averaging combines face or interface values by their areas; it ",
                    "requires a face or interface source location.",
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
            "area-weighted-average",
            "Average face or interface values weighted by area",
            "recovery = fea.RecoveryPolicy.areaWeightedAverage();",
            concat!(
                "assert(strcmp(class(recovery), 'fea.RecoveryPolicy'));\n",
                "assert(recovery.value == \"area_weighted_average\");",
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
    FEA_RECOVERY_POLICY_AREA_WEIGHTED_AVERAGE_CATALOG_ENTRY,
    name: "fea.RecoveryPolicy.areaWeightedAverage",
    result: classes::RECOVERY_POLICY,
    signature: "recovery = fea.RecoveryPolicy.areaWeightedAverage()",
    inputs: [
    ],
    output: ("recovery", "Typed fea.RecoveryPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
