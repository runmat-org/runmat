use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumAxis.periodSeconds",
    summary: "Create a response-spectrum abscissa of oscillator periods in seconds.",
    description: concat!(
        "`fea.SpectrumAxis.periodSeconds(values)` returns an `fea.SpectrumAxis` whose ",
        "`value.period_seconds` row holds oscillator natural periods, in seconds, as used by ",
        "most building-code design spectra.",
    ),
    keywords: &["fea", "response spectrum", "spectrum axis", "period seconds"],
    related: &["fea.SpectrumAxis.frequencyHertz", "fea.SpectrumInterpolation.linearPeriod", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`values` must be a nonempty real row or column vector of finite, positive, strictly ",
                    "increasing periods; the axis is stored in the order given and is not converted to ",
                    "frequency. Invalid values fail with `RunMat:fea:InvalidInput`.",
                ),
                concat!(
                    "A period axis can be interpolated with `fea.SpectrumInterpolation.linearPeriod()` or",
                    " `logLog()`; pairing it with `linearFrequency()` fails when the spectrum is created.",
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
            "design-periods",
            "Define a 0.03 to 4 s design-spectrum axis",
            "axis = fea.SpectrumAxis.periodSeconds([0.03 0.1 0.5 1 2 4]);",
            concat!(
                "assert(strcmp(class(axis), 'fea.SpectrumAxis'));\n",
                "assert(numel(axis.value.period_seconds) == 6);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Must periods be increasing even though frequencies then decrease?",
            concat!(
                "Yes. The axis is strictly increasing in its own unit; RunMat does not reorder or ",
                "convert it.",
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
    FEA_SPECTRUM_AXIS_PERIOD_SECONDS_CATALOG_ENTRY,
    name: "fea.SpectrumAxis.periodSeconds",
    result: classes::SPECTRUM_AXIS,
    signature: "axis = fea.SpectrumAxis.periodSeconds(values)",
    inputs: [
        (
            "values",
            NumericArray,
            "Positive, strictly increasing oscillator periods in seconds.",
            PolicyArg::RealVector,
        ),
    ],
    output: ("axis", "Typed fea.SpectrumAxis value."),
    integer_capabilities: &integer::SPECTRUM_AXIS_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
