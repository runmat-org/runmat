use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumConstruction.fromAccelerationHistoryExact",
    summary: concat!(
        "Record that a spectrum was derived from an acceleration history by the exact ",
        "piecewise-linear method.",
    ),
    description: concat!(
        "`fea.SpectrumConstruction.fromAccelerationHistoryExact(accelerationSignal)` returns ",
        "an `fea.SpectrumConstruction` whose `value.derived_from_acceleration_history` holds ",
        "the source signal's identifier and the method `exact_piecewise_linear`.",
    ),
    keywords: &["fea", "response spectrum", "spectrum construction", "from acceleration history exact"],
    related: &["fea.SpectrumConstruction.authoredDesign", "fea.AccelerationSignal.fromTimeKnots", "fea.SpectrumEvidence.bounded", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`accelerationSignal` must be a time-domain acceleration signal, for example from ",
                    "`fea.AccelerationSignal.fromTimeKnots`. A constant acceleration or any other ",
                    "quantity fails with `RunMat:fea:InvalidInput` (\"construction requires a time-domain ",
                    "acceleration signal\"). Only the signal identifier is stored.",
                ),
                concat!(
                    "When the model is validated, the identifier must name a time-domain acceleration ",
                    "signal registered in the same model's `Signals`.",
                ),
                concat!(
                    "The exact method integrates each oscillator exactly over every piecewise-linear ",
                    "acceleration interval and searches each interval for the peak response, honoring ",
                    "jump limits.",
                ),
                super::CONSTRUCTION_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Pass the construction as the fourteenth argument of `fea.ResponseSpectrum.new`, and ",
                    "include the source signal in `fea.model(..., \"Signals\", {...})`.",
                ),
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "exact",
            "Record a shock-pulse-derived spectrum",
            concat!(
                "pulse = fea.AccelerationSignal.fromTimeKnots('shock_pulse', {fea.SignalKnot.continuous(0, [0 0 0]), fea.SignalKnot.continuous(0.0055, [0 0 490]), fea.SignalKnot.continuous(0.011, [0 0 0])}, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.zero());\n",
                "construction = fea.SpectrumConstruction.fromAccelerationHistoryExact(pulse);",
            ),
            concat!(
                "assert(strcmp(class(construction), 'fea.SpectrumConstruction'));\n",
                "assert(construction.value.derived_from_acceleration_history.source_signal_id == \"shock_pulse\");\n",
                "assert(construction.value.derived_from_acceleration_history.method == \"exact_piecewise_linear\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Does this compute the spectrum ordinates?",
            concat!(
                "No. It records provenance. Pass the ordinates to `fea.ResponseSpectrum.new` ",
                "yourself, and register the source signal in the model.",
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
            "Construction and evidence reject invalid sources and ranges",
            "crates/runmat-runtime/src/builtins/fea/response_spectrum_policies.rs::construction_and_evidence_reject_invalid_sources_and_ranges",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPECTRUM_CONSTRUCTION_FROM_ACCELERATION_HISTORY_EXACT_CATALOG_ENTRY,
    name: "fea.SpectrumConstruction.fromAccelerationHistoryExact",
    result: classes::SPECTRUM_CONSTRUCTION,
    signature: concat!(
        "construction = ",
        "fea.SpectrumConstruction.fromAccelerationHistoryExact(accelerationSignal)",
    ),
    inputs: [
        (
            "accelerationSignal",
            Any,
            "Typed time-domain fea.AccelerationSignal the spectrum was derived from.",
            PolicyArg::Typed(&[classes::ACCELERATION_SIGNAL]),
        ),
    ],
    output: ("construction", "Typed fea.SpectrumConstruction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
