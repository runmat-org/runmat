use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DiagnosticKind.resource",
    summary: "Diagnose resource use.",
    description: concat!(
        "`fea.DiagnosticKind.resource()` returns the `fea.DiagnosticKind` whose `value` is ",
        "`resource`. Resource reports time and memory use against the step's resource ",
        "ceilings.",
    ),
    keywords: &["fea", "observation", "diagnostic kind", "resource"],
    related: &["fea.DiagnosticKind.convergence", "fea.DiagnosticKind.balance", "fea.DiagnosticKind.constraint", "fea.DiagnosticKind.contact", "fea.Observation.diagnostic"],
    sections: &[
        section(
            "Behavior",
            &[
                "Resource reports time and memory use against the step's resource ceilings.",
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
            "resource",
            "Diagnose resource use",
            "kind = fea.DiagnosticKind.resource();",
            concat!(
                "assert(strcmp(class(kind), 'fea.DiagnosticKind'));\n",
                "assert(kind.value == \"resource\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/diagnostic_kind/tests.rs::diagnostic_kind_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_DIAGNOSTIC_KIND_RESOURCE_CATALOG_ENTRY,
    name: "fea.DiagnosticKind.resource",
    result: classes::DIAGNOSTIC_KIND,
    signature: "kind = fea.DiagnosticKind.resource()",
    inputs: [
    ],
    output: ("kind", "Typed fea.DiagnosticKind value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
