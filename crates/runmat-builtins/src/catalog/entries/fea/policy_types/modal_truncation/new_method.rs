use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ModalTruncation.new",
    summary: "Require a minimum captured modal mass and choose how omitted modes are corrected.",
    description: concat!(
        "`fea.ModalTruncation.new(minimumMassRatio, correction)` returns an ",
        "`fea.ModalTruncation` with `value.minimum_cumulative_mass_ratio` and ",
        "`value.correction`: modes are retained until their cumulative effective mass reaches",
        " the ratio, and the correction accounts for the rest.",
    ),
    keywords: &["fea", "response spectrum", "modal truncation", "new"],
    related: &["fea.ModalTruncationCorrection.missingMass", "fea.ModalTruncationCorrection.none", "fea.ModalTruncationCorrection.residualMode", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`minimumMassRatio` must be a finite fraction in [0, 1]; 0.9 is the common 90 percent",
                    " requirement. `correction` must be a typed `fea.ModalTruncationCorrection`. Invalid ",
                    "inputs fail with `RunMat:fea:InvalidInput`.",
                ),
                concat!(
                    "RunMat's modal response-spectrum steps also carry a whole-result accuracy criterion ",
                    "on modal mass capture (relative tolerance 0.1 in the step's default plan).",
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
            "ninety-percent",
            "Capture 90 percent of mass with a missing-mass correction",
            concat!(
                "correction = fea.ModalTruncationCorrection.missingMass(4.9);\n",
                "truncation = fea.ModalTruncation.new(0.9, correction);",
            ),
            concat!(
                "assert(strcmp(class(truncation), 'fea.ModalTruncation'));\n",
                "assert(truncation.value.minimum_cumulative_mass_ratio == 0.9);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is the ratio a percentage?",
            "No. Write 0.9 for 90 percent.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/modal_truncation/tests.rs::modal_truncation_identities_conform",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_MODAL_TRUNCATION_NEW_CATALOG_ENTRY,
    name: "fea.ModalTruncation.new",
    result: classes::MODAL_TRUNCATION,
    signature: "policy = fea.ModalTruncation.new(minimumMassRatio, correction)",
    inputs: [
        (
            "minimumMassRatio",
            NumericScalar,
            "Required cumulative effective modal-mass fraction in [0, 1], for example 0.9.",
            PolicyArg::RealScalar,
        ),
        (
            "correction",
            Any,
            "Typed fea.ModalTruncationCorrection for the response of omitted modes.",
            PolicyArg::Typed(&[classes::MODAL_TRUNCATION_CORRECTION]),
        ),
    ],
    output: ("policy", "Typed fea.ModalTruncation value."),
    integer_capabilities: &integer::TRUNCATION_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
