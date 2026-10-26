use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumDamping.ratio",
    summary: concat!(
        "State the oscillator damping of a response spectrum as a fraction of critical ",
        "damping.",
    ),
    description: concat!(
        "`fea.SpectrumDamping.ratio(value)` returns an `fea.SpectrumDamping` whose ",
        "`value.ratio` is the single-degree-of-freedom damping ratio the spectrum was ",
        "computed for.",
    ),
    keywords: &["fea", "response spectrum", "spectrum damping", "ratio"],
    related: &["fea.SpectrumDamping.qualityFactor", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`value` must be a finite real scalar with 0 <= value < 1; 0.05 means 5 percent of ",
                    "critical damping. Values outside that range fail with `RunMat:fea:InvalidInput`.",
                ),
                concat!(
                    "Damping describes the spectrum's oscillators; it does not add damping to the ",
                    "structural model.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "five-percent",
            "Use the usual 5 percent damped spectrum",
            "damping = fea.SpectrumDamping.ratio(0.05);",
            concat!(
                "assert(strcmp(class(damping), 'fea.SpectrumDamping'));\n",
                "assert(damping.value.ratio == 0.05);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is 5 the same as 0.05?",
            "No. The ratio is a fraction; 5 is rejected because it is not less than 1.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/spectrum_damping/tests.rs::spectrum_damping_identities_conform",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPECTRUM_DAMPING_RATIO_CATALOG_ENTRY,
    name: "fea.SpectrumDamping.ratio",
    result: classes::SPECTRUM_DAMPING,
    signature: "damping = fea.SpectrumDamping.ratio(value)",
    inputs: [
        (
            "value",
            NumericScalar,
            "Fraction of critical damping, at least 0 and less than 1 (0.05 is 5 percent).",
            PolicyArg::RealScalar,
        ),
    ],
    output: ("damping", "Typed fea.SpectrumDamping value."),
    integer_capabilities: &integer::SPECTRUM_SCALAR_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
