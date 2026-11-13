use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ComplexRepresentation.real",
    summary: "Report the real part of a result.",
    description: concat!(
        "`fea.ComplexRepresentation.real()` returns the `fea.ComplexRepresentation` whose ",
        "`value` is `real`. Real is required for procedures with real-valued results (static,",
        " transient, modal, and so on) and selects the real part of complex harmonic or ",
        "random-response results.",
    ),
    keywords: &["fea", "observation", "complex representation", "real"],
    related: &["fea.ComplexRepresentation.imaginary", "fea.ComplexRepresentation.magnitude", "fea.ComplexRepresentation.phaseRadians", "fea.ComplexRepresentation.complexValue", "fea.SignalPrecision.complex128", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Real is required for procedures with real-valued results (static, transient, modal, ",
                    "and so on) and selects the real part of complex harmonic or random-response results.",
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
            "real",
            "Report the real part of a result",
            "representation = fea.ComplexRepresentation.real();",
            concat!(
                "assert(strcmp(class(representation), 'fea.ComplexRepresentation'));\n",
                "assert(representation.value == \"real\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/complex_representation/tests.rs::complex_representation_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_COMPLEX_REPRESENTATION_REAL_CATALOG_ENTRY,
    name: "fea.ComplexRepresentation.real",
    result: classes::COMPLEX_REPRESENTATION,
    signature: "representation = fea.ComplexRepresentation.real()",
    inputs: [
    ],
    output: ("representation", "Typed fea.ComplexRepresentation value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
