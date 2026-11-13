use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DiagnosticKind.convergence",
    summary: "Diagnose solver convergence.",
    description: concat!(
        "`fea.DiagnosticKind.convergence()` returns the `fea.DiagnosticKind` whose `value` is",
        " `convergence`. Convergence reports iteration and convergence history of the solver.",
    ),
    keywords: &["fea", "observation", "diagnostic kind", "convergence"],
    related: &["fea.DiagnosticKind.balance", "fea.DiagnosticKind.constraint", "fea.DiagnosticKind.contact", "fea.DiagnosticKind.adaptation", "fea.Observation.diagnostic"],
    sections: &[
        section(
            "Behavior",
            &[
                "Convergence reports iteration and convergence history of the solver.",
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
            "convergence",
            "Diagnose solver convergence",
            "kind = fea.DiagnosticKind.convergence();",
            concat!(
                "assert(strcmp(class(kind), 'fea.DiagnosticKind'));\n",
                "assert(kind.value == \"convergence\");",
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
    FEA_DIAGNOSTIC_KIND_CONVERGENCE_CATALOG_ENTRY,
    name: "fea.DiagnosticKind.convergence",
    result: classes::DIAGNOSTIC_KIND,
    signature: "kind = fea.DiagnosticKind.convergence()",
    inputs: [
    ],
    output: ("kind", "Typed fea.DiagnosticKind value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
