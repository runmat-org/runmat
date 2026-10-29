use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumEvidence.bounded",
    summary: "Record sampling adequacy and an interpolation-error bound for a response spectrum.",
    description: concat!(
        "`fea.SpectrumEvidence.bounded(samplingAdequacyRatio, maximumInterpolationError)` ",
        "returns an `fea.SpectrumEvidence` with `value.sampling_adequacy_ratio` and ",
        "`value.max_interpolation_error`.",
    ),
    keywords: &["fea", "response spectrum", "spectrum evidence", "bounded"],
    related: &["fea.SpectrumConstruction.authoredDesign", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`samplingAdequacyRatio` must be a finite fraction in [0, 1], and ",
                    "`maximumInterpolationError` must be finite and nonnegative, in the unit of the ",
                    "spectrum ordinates. Invalid values fail with `RunMat:fea:InvalidInput`.",
                ),
                concat!(
                    "For spectra RunMat derives itself, the sampling adequacy is the number of source ",
                    "samples per cycle of the highest abscissa frequency divided by 20, capped at 1.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                "Pass the evidence as the fifteenth argument of `fea.ResponseSpectrum.new`.",
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "bounded",
            "Record 95 percent sampling adequacy",
            "evidence = fea.SpectrumEvidence.bounded(0.95, 0.02);",
            concat!(
                "assert(strcmp(class(evidence), 'fea.SpectrumEvidence'));\n",
                "assert(evidence.value.sampling_adequacy_ratio == 0.95);\n",
                "assert(evidence.value.max_interpolation_error == 0.02);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "What does a ratio of 1 mean?",
            concat!(
                "At least 20 source samples per cycle of the highest spectrum frequency, the level ",
                "RunMat treats as fully adequate.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/spectrum_evidence/tests.rs::spectrum_evidence_identities_conform",
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
    FEA_SPECTRUM_EVIDENCE_BOUNDED_CATALOG_ENTRY,
    name: "fea.SpectrumEvidence.bounded",
    result: classes::SPECTRUM_EVIDENCE,
    signature: concat!(
        "evidence = fea.SpectrumEvidence.bounded(samplingAdequacyRatio, ",
        "maximumInterpolationError)",
    ),
    inputs: [
        (
            "samplingAdequacyRatio",
            NumericScalar,
            "Fraction in [0, 1] describing how adequately the source was sampled.",
            PolicyArg::RealScalar,
        ),
        (
            "maximumInterpolationError",
            NumericScalar,
            "Nonnegative bound on interpolation error, in the ordinate unit.",
            PolicyArg::RealScalar,
        ),
    ],
    output: ("evidence", "Typed fea.SpectrumEvidence value."),
    integer_capabilities: &integer::SPECTRUM_EVIDENCE_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
