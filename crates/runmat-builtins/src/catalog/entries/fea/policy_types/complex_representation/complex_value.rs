use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ComplexRepresentation.complexValue",
    summary: "Report the full complex value.",
    description: concat!(
        "`fea.ComplexRepresentation.complexValue()` returns the `fea.ComplexRepresentation` ",
        "whose `value` is `complex`. Complex keeps both parts.",
    ),
    keywords: &["fea", "observation", "complex representation", "complex value"],
    related: &["fea.ComplexRepresentation.real", "fea.ComplexRepresentation.imaginary", "fea.ComplexRepresentation.magnitude", "fea.ComplexRepresentation.phaseRadians", "fea.SignalPrecision.complex128", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Complex keeps both parts. It requires a complex procedure and complex storage ",
                    "precision (`fea.SignalPrecision.complex64()` or `complex128()`); every other ",
                    "representation requires real storage.",
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
            "complex-value",
            "Report the full complex value",
            "representation = fea.ComplexRepresentation.complexValue();",
            concat!(
                "assert(strcmp(class(representation), 'fea.ComplexRepresentation'));\n",
                "assert(representation.value == \"complex\");",
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
    FEA_COMPLEX_REPRESENTATION_COMPLEX_VALUE_CATALOG_ENTRY,
    name: "fea.ComplexRepresentation.complexValue",
    result: classes::COMPLEX_REPRESENTATION,
    signature: "representation = fea.ComplexRepresentation.complexValue()",
    inputs: [
    ],
    output: ("representation", "Typed fea.ComplexRepresentation value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
