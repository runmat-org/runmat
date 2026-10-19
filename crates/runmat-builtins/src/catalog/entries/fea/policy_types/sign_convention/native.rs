use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignConvention.native",
    summary: "Report values in the formulation's own sign convention.",
    description: concat!(
        "`fea.SignConvention.native()` returns the `fea.SignConvention` whose `value` is ",
        "`native`. Native keeps the sign the solver produces and is valid for every quantity.",
    ),
    keywords: &["fea", "observation", "sign convention", "native"],
    related: &["fea.SignConvention.tensionPositive", "fea.SignConvention.compressionPositive", "fea.SignConvention.outwardPositive", "fea.SignConvention.inwardPositive", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                "Native keeps the sign the solver produces and is valid for every quantity.",
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
            "Report values in the formulation's own sign convention",
            "sign = fea.SignConvention.native();",
            concat!(
                "assert(strcmp(class(sign), 'fea.SignConvention'));\n",
                "assert(sign.value == \"native\");",
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
    FEA_SIGN_CONVENTION_NATIVE_CATALOG_ENTRY,
    name: "fea.SignConvention.native",
    result: classes::SIGN_CONVENTION,
    signature: "sign = fea.SignConvention.native()",
    inputs: [
    ],
    output: ("sign", "Typed fea.SignConvention value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
