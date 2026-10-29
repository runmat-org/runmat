use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ResponseSpectrum.new",
    summary: "Create an immutable, typed response spectrum for modal response-spectrum analysis.",
    description: concat!(
        "`fea.ResponseSpectrum.new(id, axis, ordinates, damping, response, envelope, ",
        "interpolation, extrapolation, direction, modalCombination, directionalCombination, ",
        "truncation, rigidResponse, construction, evidence, provenanceDigest)` returns an ",
        "`fea.ResponseSpectrum` whose properties are `spectrum_id`, `abscissa`, `ordinates`, ",
        "`damping`, `response`, `envelope`, `interpolation`, `extrapolation`, `direction`, ",
        "`modal_combination`, `directional_combination`, `truncation`, `rigid_response`, ",
        "`construction`, and `evidence`.",
    ),
    keywords: &["fea", "response spectrum", "seismic", "shock", "modal", "CQC"],
    related: &["fea.SpectrumAxis.frequencyHertz", "fea.SpectrumDamping.ratio", "fea.SpectrumResponse.pseudoAcceleration", "fea.ModalCombination.completeQuadratic", "fea.ModalTruncation.new", "fea.RigidResponse.gupta", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "All sixteen arguments are required. Each policy argument must be a value of its ",
                    "typed class, `id` and `provenanceDigest` must be nonempty text, and `ordinates` must",
                    " be a finite real vector. The call fails with `RunMat:fea:InvalidInput` and returns ",
                    "nothing when any check fails.",
                ),
                concat!(
                    "The assembled spectrum is then checked as a whole: the axis and ordinates must have ",
                    "the same length, ordinates must be nonnegative (strictly positive with `logLog` ",
                    "interpolation), and the interpolation must match the axis (`linearFrequency` with a ",
                    "frequency axis, `linearPeriod` with a period axis). Missing-mass correction requires",
                    " an `absoluteAcceleration` or `pseudoAcceleration` response, and any rigid-response ",
                    "policy other than `disabled` requires an acceleration response and CQC modal ",
                    "combination.",
                ),
                concat!(
                    "Ordinates are in the canonical SI unit of the response type: meters for relative ",
                    "displacement, m/s for relative and pseudo velocity, and m/s^2 for accelerations. The",
                    " spectrum stores them with the provenance digest as authored numerical data. Integer",
                    " ordinate arrays are accepted and converted once to binary64.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                super::SPECTRUM_STUDY,
                concat!(
                    "A spectrum constructed from an acceleration history must have that history in the ",
                    "same model's `Signals`; model validation checks that the named signal exists and is ",
                    "a time-domain acceleration signal.",
                ),
            ],
        ),
    ],
    examples: &[
        example(
            "site-spectrum",
            "Author a 5 percent damped site design spectrum",
            concat!(
                "axis = fea.SpectrumAxis.frequencyHertz([1 2 5 10 33 100]);\n",
                "direction = fea.SpectrumDirection.new(fea.CoordinateFrame.globalFrame(), [1 0 0]);\n",
                "truncation = fea.ModalTruncation.new(0.9, fea.ModalTruncationCorrection.missingMass(3.9));\n",
                "spectrum = fea.ResponseSpectrum.new('site_x', axis, 9.81 * [0.2 0.5 1.0 1.0 0.4 0.4], ...\n",
                "    fea.SpectrumDamping.ratio(0.05), fea.SpectrumResponse.pseudoAcceleration(), ...\n",
                "    fea.SpectrumEnvelope.maximax(), fea.SpectrumInterpolation.logLog(), ...\n",
                "    fea.SpectrumExtrapolation.holdEndpoint(), direction, ...\n",
                "    fea.ModalCombination.completeQuadratic(), fea.DirectionalCombination.srss(), ...\n",
                "    truncation, fea.RigidResponse.gupta(8, 33), ...\n",
                "    fea.SpectrumConstruction.authoredDesign(), fea.SpectrumEvidence.bounded(1, 0), ...\n",
                "    'site-report-2026-rev3');",
            ),
            concat!(
                "assert(strcmp(class(spectrum), 'fea.ResponseSpectrum'));\n",
                "assert(spectrum.spectrum_id == \"site_x\");\n",
                "assert(spectrum.damping.ratio == 0.05);\n",
                "assert(spectrum.modal_combination == \"complete_quadratic_combination\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Why sixteen positional arguments?",
            concat!(
                "Every choice that changes the analysis result is explicit and typed, so two spectra ",
                "with equal ordinates but different damping, combination, or truncation can never be ",
                "confused.",
            ),
        ),
        faq(
            "Are ordinates in g?",
            concat!(
                "No. Accelerations are in m/s^2; multiply g values by 9.81 (or your local standard ",
                "gravity).",
            ),
        ),
    ],
    links: &[
        page("Build a modal response-spectrum study", "/docs/simulation/language/guides/modal-and-response-spectrum-study"),
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/response_spectra.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/response_spectrum/tests.rs::response_spectrum_identity_conforms",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_RESPONSE_SPECTRUM_NEW_CATALOG_ENTRY,
    name: "fea.ResponseSpectrum.new",
    result: classes::RESPONSE_SPECTRUM,
    signature: concat!(
        "spectrum = fea.ResponseSpectrum.new(id, axis, ordinates, damping, response, ",
        "envelope, interpolation, extrapolation, direction, modalCombination, ",
        "directionalCombination, truncation, rigidResponse, construction, evidence, ",
        "provenanceDigest)",
    ),
    inputs: [
        (
            "id",
            StringScalar,
            "Nonempty spectrum identifier, unique within the model.",
            PolicyArg::Text,
        ),
        (
            "axis",
            Any,
            "Typed fea.SpectrumAxis (frequency in Hz or period in s).",
            PolicyArg::Typed(&[classes::SPECTRUM_AXIS]),
        ),
        (
            "ordinates",
            NumericArray,
            "Finite, nonnegative ordinates, one per abscissa, in the unit of the response type.",
            PolicyArg::RealVector,
        ),
        (
            "damping",
            Any,
            "Typed fea.SpectrumDamping.",
            PolicyArg::Typed(&[classes::SPECTRUM_DAMPING]),
        ),
        (
            "response",
            Any,
            "Typed fea.SpectrumResponse (what the ordinates measure).",
            PolicyArg::Typed(&[classes::SPECTRUM_RESPONSE]),
        ),
        (
            "envelope",
            Any,
            "Typed fea.SpectrumEnvelope.",
            PolicyArg::Typed(&[classes::SPECTRUM_ENVELOPE]),
        ),
        (
            "interpolation",
            Any,
            "Typed fea.SpectrumInterpolation; must match the axis.",
            PolicyArg::Typed(&[classes::SPECTRUM_INTERPOLATION]),
        ),
        (
            "extrapolation",
            Any,
            "Typed fea.SpectrumExtrapolation outside the axis range.",
            PolicyArg::Typed(&[classes::SPECTRUM_EXTRAPOLATION]),
        ),
        (
            "direction",
            Any,
            "Typed fea.SpectrumDirection of the base excitation.",
            PolicyArg::Typed(&[classes::SPECTRUM_DIRECTION]),
        ),
        (
            "modalCombination",
            Any,
            "Typed fea.ModalCombination rule.",
            PolicyArg::Typed(&[classes::MODAL_COMBINATION]),
        ),
        (
            "directionalCombination",
            Any,
            "Typed fea.DirectionalCombination rule.",
            PolicyArg::Typed(&[classes::DIRECTIONAL_COMBINATION]),
        ),
        (
            "truncation",
            Any,
            "Typed fea.ModalTruncation policy.",
            PolicyArg::Typed(&[classes::MODAL_TRUNCATION]),
        ),
        (
            "rigidResponse",
            Any,
            "Typed fea.RigidResponse policy.",
            PolicyArg::Typed(&[classes::RIGID_RESPONSE]),
        ),
        (
            "construction",
            Any,
            "Typed fea.SpectrumConstruction provenance.",
            PolicyArg::Typed(&[classes::SPECTRUM_CONSTRUCTION]),
        ),
        (
            "evidence",
            Any,
            "Typed fea.SpectrumEvidence.",
            PolicyArg::Typed(&[classes::SPECTRUM_EVIDENCE]),
        ),
        (
            "provenanceDigest",
            StringScalar,
            concat!(
                "Nonempty provenance text for the ordinates, such as a document reference or content ",
                "digest.",
            ),
            PolicyArg::Text,
        ),
    ],
    output: ("spectrum", "Typed, immutable fea.ResponseSpectrum."),
    integer_capabilities: &integer::RESPONSE_SPECTRUM_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
