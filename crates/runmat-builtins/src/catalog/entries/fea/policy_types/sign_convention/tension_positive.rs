use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignConvention.tensionPositive",
    summary: "Count tension as positive.",
    description: concat!(
        "`fea.SignConvention.tensionPositive()` returns the `fea.SignConvention` whose ",
        "`value` is `tension_positive`. Tension-positive is valid for strain, stress, and ",
        "contact pressure.",
    ),
    keywords: &["fea", "observation", "sign convention", "tension positive"],
    related: &["fea.SignConvention.native", "fea.SignConvention.compressionPositive", "fea.SignConvention.outwardPositive", "fea.SignConvention.inwardPositive", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                "Tension-positive is valid for strain, stress, and contact pressure.",
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
            "tension-positive",
            "Count tension as positive",
            "sign = fea.SignConvention.tensionPositive();",
            concat!(
                "assert(strcmp(class(sign), 'fea.SignConvention'));\n",
                "assert(sign.value == \"tension_positive\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/sign_convention/tests.rs::sign_convention_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SIGN_CONVENTION_TENSION_POSITIVE_CATALOG_ENTRY,
    name: "fea.SignConvention.tensionPositive",
    result: classes::SIGN_CONVENTION,
    signature: "sign = fea.SignConvention.tensionPositive()",
    inputs: [
    ],
    output: ("sign", "Typed fea.SignConvention value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
