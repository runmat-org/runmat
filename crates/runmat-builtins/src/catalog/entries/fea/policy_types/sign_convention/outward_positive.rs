use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignConvention.outwardPositive",
    summary: "Count flow out of the region as positive.",
    description: concat!(
        "`fea.SignConvention.outwardPositive()` returns the `fea.SignConvention` whose ",
        "`value` is `outward_positive`. Outward-positive is valid for heat flux, heat rate, ",
        "electric current, current density, mass flow rate, acoustic particle velocity, and ",
        "fluid velocity.",
    ),
    keywords: &["fea", "observation", "sign convention", "outward positive"],
    related: &["fea.SignConvention.native", "fea.SignConvention.tensionPositive", "fea.SignConvention.compressionPositive", "fea.SignConvention.inwardPositive", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Outward-positive is valid for heat flux, heat rate, electric current, current ",
                    "density, mass flow rate, acoustic particle velocity, and fluid velocity.",
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
            "outward-positive",
            "Count flow out of the region as positive",
            "sign = fea.SignConvention.outwardPositive();",
            concat!(
                "assert(strcmp(class(sign), 'fea.SignConvention'));\n",
                "assert(sign.value == \"outward_positive\");",
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
    FEA_SIGN_CONVENTION_OUTWARD_POSITIVE_CATALOG_ENTRY,
    name: "fea.SignConvention.outwardPositive",
    result: classes::SIGN_CONVENTION,
    signature: "sign = fea.SignConvention.outwardPositive()",
    inputs: [
    ],
    output: ("sign", "Typed fea.SignConvention value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
