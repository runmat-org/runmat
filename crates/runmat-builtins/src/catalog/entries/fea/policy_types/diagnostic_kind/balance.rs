use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DiagnosticKind.balance",
    summary: "Diagnose conservation balance.",
    description: concat!(
        "`fea.DiagnosticKind.balance()` returns the `fea.DiagnosticKind` whose `value` is ",
        "`balance`. Balance reports how well conserved quantities balance across the model.",
    ),
    keywords: &["fea", "observation", "diagnostic kind", "balance"],
    related: &["fea.DiagnosticKind.convergence", "fea.DiagnosticKind.constraint", "fea.DiagnosticKind.contact", "fea.DiagnosticKind.adaptation", "fea.Observation.diagnostic"],
    sections: &[
        section(
            "Behavior",
            &[
                "Balance reports how well conserved quantities balance across the model.",
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
            "balance",
            "Diagnose conservation balance",
            "kind = fea.DiagnosticKind.balance();",
            concat!(
                "assert(strcmp(class(kind), 'fea.DiagnosticKind'));\n",
                "assert(kind.value == \"balance\");",
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
    FEA_DIAGNOSTIC_KIND_BALANCE_CATALOG_ENTRY,
    name: "fea.DiagnosticKind.balance",
    result: classes::DIAGNOSTIC_KIND,
    signature: "kind = fea.DiagnosticKind.balance()",
    inputs: [
    ],
    output: ("kind", "Typed fea.DiagnosticKind value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
