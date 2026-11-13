use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RecoveryPolicy.extrapolate",
    summary: "Extrapolate integration-point or element values to the output location.",
    description: concat!(
        "`fea.RecoveryPolicy.extrapolate()` returns the `fea.RecoveryPolicy` whose `value` is",
        " `extrapolate`. Extrapolation uses the element's own interpolation to carry ",
        "integration-point, element, or cell values to nodes or other output locations, ",
        "without averaging across elements.",
    ),
    keywords: &["fea", "observation", "recovery policy", "extrapolate"],
    related: &["fea.RecoveryPolicy.native", "fea.RecoveryPolicy.volumeWeightedAverage", "fea.RecoveryPolicy.areaWeightedAverage", "fea.RecoveryPolicy.conservativeProjection", "fea.FieldLocation.node", "fea.FieldLocation.integrationPoint"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Extrapolation uses the element's own interpolation to carry integration-point, ",
                    "element, or cell values to nodes or other output locations, without averaging across",
                    " elements.",
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
            "extrapolate",
            "Extrapolate integration-point or element values to the output location",
            "recovery = fea.RecoveryPolicy.extrapolate();",
            concat!(
                "assert(strcmp(class(recovery), 'fea.RecoveryPolicy'));\n",
                "assert(recovery.value == \"extrapolate\");",
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
    FEA_RECOVERY_POLICY_EXTRAPOLATE_CATALOG_ENTRY,
    name: "fea.RecoveryPolicy.extrapolate",
    result: classes::RECOVERY_POLICY,
    signature: "recovery = fea.RecoveryPolicy.extrapolate()",
    inputs: [
    ],
    output: ("recovery", "Typed fea.RecoveryPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
