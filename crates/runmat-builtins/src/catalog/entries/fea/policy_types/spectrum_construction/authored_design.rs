use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumConstruction.authoredDesign",
    summary: "Record that a spectrum's ordinates are an authored design spectrum.",
    description: concat!(
        "`fea.SpectrumConstruction.authoredDesign()` returns the `fea.SpectrumConstruction` ",
        "whose `value` is `authored_design`: the ordinates were written by the author, for ",
        "example from a design code or a test specification.",
    ),
    keywords: &["fea", "response spectrum", "spectrum construction", "authored design"],
    related: &["fea.SpectrumConstruction.fromAccelerationHistoryExact", "fea.SpectrumConstruction.fromAccelerationHistoryRecursive", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                "The constructor takes no arguments.",
                super::CONSTRUCTION_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                "Pass the construction as the fourteenth argument of `fea.ResponseSpectrum.new`.",
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "design",
            "Mark a code design spectrum as authored",
            "construction = fea.SpectrumConstruction.authoredDesign();",
            concat!(
                "assert(strcmp(class(construction), 'fea.SpectrumConstruction'));\n",
                "assert(construction.value == \"authored_design\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "When should I use authored design?",
            concat!(
                "Whenever the ordinates come from a document or a specification rather than from an ",
                "acceleration history in the same model.",
            ),
        ),
    ],
    links: &[
        page("Build a modal response-spectrum study", "/docs/simulation/language/guides/modal-and-response-spectrum-study"),
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/response_spectrum_policies/parameters.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/spectrum_construction/tests.rs::spectrum_construction_identities_conform",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPECTRUM_CONSTRUCTION_AUTHORED_DESIGN_CATALOG_ENTRY,
    name: "fea.SpectrumConstruction.authoredDesign",
    result: classes::SPECTRUM_CONSTRUCTION,
    signature: "construction = fea.SpectrumConstruction.authoredDesign()",
    inputs: [
    ],
    output: ("construction", "Typed fea.SpectrumConstruction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
