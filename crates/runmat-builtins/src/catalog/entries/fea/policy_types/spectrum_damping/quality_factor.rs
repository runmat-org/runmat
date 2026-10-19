use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumDamping.qualityFactor",
    summary: "State the oscillator damping of a response spectrum as a quality factor Q.",
    description: concat!(
        "`fea.SpectrumDamping.qualityFactor(value)` returns an `fea.SpectrumDamping` whose ",
        "`value.quality_factor` is Q, the convention of shock-response specifications (Q = 10",
        " corresponds to a 5 percent damping ratio, Q = 1 / (2 * ratio)).",
    ),
    keywords: &["fea", "response spectrum", "spectrum damping", "quality factor"],
    related: &["fea.SpectrumDamping.ratio", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`value` must be a finite, positive real scalar; zero or negative values fail with ",
                    "`RunMat:fea:InvalidInput`. The value is stored as Q and is not converted to a ratio.",
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
            "q-ten",
            "Use Q = 10 for a shock specification",
            "damping = fea.SpectrumDamping.qualityFactor(10);",
            concat!(
                "assert(strcmp(class(damping), 'fea.SpectrumDamping'));\n",
                "assert(damping.value.quality_factor == 10);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Which damping ratio does Q = 10 represent?",
            "5 percent, since the ratio is 1 / (2Q).",
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
    FEA_SPECTRUM_DAMPING_QUALITY_FACTOR_CATALOG_ENTRY,
    name: "fea.SpectrumDamping.qualityFactor",
    result: classes::SPECTRUM_DAMPING,
    signature: "damping = fea.SpectrumDamping.qualityFactor(value)",
    inputs: [
        (
            "value",
            NumericScalar,
            "Positive amplification factor Q of the spectrum oscillators.",
            PolicyArg::RealScalar,
        ),
    ],
    output: ("damping", "Typed fea.SpectrumDamping value."),
    integer_capabilities: &integer::SPECTRUM_SCALAR_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
