use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DiagnosticKind.constraint",
    summary: "Diagnose constraint satisfaction.",
    description: concat!(
        "`fea.DiagnosticKind.constraint()` returns the `fea.DiagnosticKind` whose `value` is ",
        "`constraint`. Constraint reports the state and violation of constraints and ",
        "couplings.",
    ),
    keywords: &["fea", "observation", "diagnostic kind", "constraint"],
    related: &["fea.DiagnosticKind.convergence", "fea.DiagnosticKind.balance", "fea.DiagnosticKind.contact", "fea.DiagnosticKind.adaptation", "fea.Observation.diagnostic"],
    sections: &[
        section(
            "Behavior",
            &[
                "Constraint reports the state and violation of constraints and couplings.",
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
            "constraint",
            "Diagnose constraint satisfaction",
            "kind = fea.DiagnosticKind.constraint();",
            concat!(
                "assert(strcmp(class(kind), 'fea.DiagnosticKind'));\n",
                "assert(kind.value == \"constraint\");",
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
    FEA_DIAGNOSTIC_KIND_CONSTRAINT_CATALOG_ENTRY,
    name: "fea.DiagnosticKind.constraint",
    result: classes::DIAGNOSTIC_KIND,
    signature: "kind = fea.DiagnosticKind.constraint()",
    inputs: [
    ],
    output: ("kind", "Typed fea.DiagnosticKind value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
