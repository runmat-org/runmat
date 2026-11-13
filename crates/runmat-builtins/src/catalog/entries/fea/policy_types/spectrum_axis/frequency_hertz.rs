use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumAxis.frequencyHertz",
    summary: "Create a response-spectrum abscissa of natural frequencies in hertz.",
    description: concat!(
        "`fea.SpectrumAxis.frequencyHertz(values)` returns an `fea.SpectrumAxis` whose ",
        "`value.frequency_hertz` row holds the oscillator natural frequencies, in hertz, at ",
        "which the spectrum ordinates are given.",
    ),
    keywords: &["fea", "response spectrum", "spectrum axis", "frequency hertz"],
    related: &["fea.SpectrumAxis.periodSeconds", "fea.SpectrumInterpolation.linearFrequency", "fea.SpectrumInterpolation.logLog", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`values` must be a nonempty real row or column vector of finite, positive, strictly ",
                    "increasing frequencies. Zero, negative, repeated, or decreasing values fail with ",
                    "`RunMat:fea:InvalidInput`. Integer arrays are accepted and converted once to ",
                    "binary64.",
                ),
                concat!(
                    "A frequency axis can be interpolated with ",
                    "`fea.SpectrumInterpolation.linearFrequency()` or `logLog()`; pairing it with ",
                    "`linearPeriod()` fails when the spectrum is created.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Pass the axis as the second argument of `fea.ResponseSpectrum.new`; the ordinates ",
                    "must have one value per abscissa.",
                ),
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "seismic-frequencies",
            "Define a 1 to 100 Hz spectrum axis",
            "axis = fea.SpectrumAxis.frequencyHertz([1 2 5 10 33 100]);",
            concat!(
                "assert(strcmp(class(axis), 'fea.SpectrumAxis'));\n",
                "assert(numel(axis.value.frequency_hertz) == 6);\n",
                "assert(axis.value.frequency_hertz(6) == 100);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Can the axis include 0 Hz?",
            concat!(
                "No. Every abscissa must be positive. Represent the rigid (zero-period) level with a ",
                "rigid-response policy or missing-mass correction instead.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/spectrum_axis/tests.rs::spectrum_axis_identities_conform",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPECTRUM_AXIS_FREQUENCY_HERTZ_CATALOG_ENTRY,
    name: "fea.SpectrumAxis.frequencyHertz",
    result: classes::SPECTRUM_AXIS,
    signature: "axis = fea.SpectrumAxis.frequencyHertz(values)",
    inputs: [
        (
            "values",
            NumericArray,
            "Positive, strictly increasing natural frequencies in hertz.",
            PolicyArg::RealVector,
        ),
    ],
    output: ("axis", "Typed fea.SpectrumAxis value."),
    integer_capabilities: &integer::SPECTRUM_AXIS_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
