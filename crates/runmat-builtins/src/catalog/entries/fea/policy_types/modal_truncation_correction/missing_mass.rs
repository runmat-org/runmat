use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ModalTruncationCorrection.missingMass",
    summary: concat!(
        "Correct for omitted high-frequency modes with a missing-mass term at an explicit ",
        "zero-period acceleration.",
    ),
    description: concat!(
        "`fea.ModalTruncationCorrection.missingMass(zeroPeriodAcceleration)` returns an ",
        "`fea.ModalTruncationCorrection` whose `value.missing_mass.zero_period_acceleration` ",
        "(ZPA) drives the static response of the mass not captured by the retained modes.",
    ),
    keywords: &["fea", "response spectrum", "modal truncation correction", "missing mass"],
    related: &["fea.ModalTruncation.new", "fea.ModalTruncationCorrection.none", "fea.ModalTruncationCorrection.residualMode", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The ZPA must be a finite, nonnegative real scalar in m/s^2; a negative value fails ",
                    "with `RunMat:fea:InvalidInput`.",
                ),
                concat!(
                    "When the spectrum is created, missing-mass correction requires an acceleration ",
                    "spectrum (`absoluteAcceleration` or `pseudoAcceleration` response).",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                "Pass the correction as the second argument of `fea.ModalTruncation.new`.",
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "half-g",
            "Use a 0.5 g zero-period acceleration",
            "correction = fea.ModalTruncationCorrection.missingMass(4.9);",
            concat!(
                "assert(strcmp(class(correction), 'fea.ModalTruncationCorrection'));\n",
                "assert(correction.value.missing_mass.zero_period_acceleration == 4.9);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is the ZPA in g?",
            "No. Write it in m/s^2; 0.5 g is about 4.9.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/modal_truncation_correction/tests.rs::missing_mass_correction_identity_conforms",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_MODAL_TRUNCATION_CORRECTION_MISSING_MASS_CATALOG_ENTRY,
    name: "fea.ModalTruncationCorrection.missingMass",
    result: classes::MODAL_TRUNCATION_CORRECTION,
    signature: "correction = fea.ModalTruncationCorrection.missingMass(zeroPeriodAcceleration)",
    inputs: [
        (
            "zeroPeriodAcceleration",
            NumericScalar,
            "Nonnegative zero-period acceleration in m/s^2.",
            PolicyArg::RealScalar,
        ),
    ],
    output: ("correction", "Typed fea.ModalTruncationCorrection value."),
    integer_capabilities: &integer::SPECTRUM_SCALAR_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
