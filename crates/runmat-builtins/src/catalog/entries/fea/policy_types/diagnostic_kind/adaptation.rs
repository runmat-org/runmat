use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DiagnosticKind.adaptation",
    summary: "Diagnose adaptive refinement.",
    description: concat!(
        "`fea.DiagnosticKind.adaptation()` returns the `fea.DiagnosticKind` whose `value` is ",
        "`adaptation`. Adaptation reports adaptive mesh and sampling refinement decisions.",
    ),
    keywords: &["fea", "observation", "diagnostic kind", "adaptation"],
    related: &["fea.DiagnosticKind.convergence", "fea.DiagnosticKind.balance", "fea.DiagnosticKind.constraint", "fea.DiagnosticKind.contact", "fea.Observation.diagnostic"],
    sections: &[
        section(
            "Behavior",
            &[
                "Adaptation reports adaptive mesh and sampling refinement decisions.",
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
            "adaptation",
            "Diagnose adaptive refinement",
            "kind = fea.DiagnosticKind.adaptation();",
            concat!(
                "assert(strcmp(class(kind), 'fea.DiagnosticKind'));\n",
                "assert(kind.value == \"adaptation\");",
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
    FEA_DIAGNOSTIC_KIND_ADAPTATION_CATALOG_ENTRY,
    name: "fea.DiagnosticKind.adaptation",
    result: classes::DIAGNOSTIC_KIND,
    signature: "kind = fea.DiagnosticKind.adaptation()",
    inputs: [
    ],
    output: ("kind", "Typed fea.DiagnosticKind value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
