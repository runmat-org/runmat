use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpectrumDirection.new",
    summary: "Create the excitation direction of a response spectrum.",
    description: concat!(
        "`fea.SpectrumDirection.new(frame, vector)` returns an `fea.SpectrumDirection` with ",
        "`value.frame` and a unit-length `value.vector` along which the spectrum excites the ",
        "base.",
    ),
    keywords: &["fea", "response spectrum", "spectrum direction", "new"],
    related: &["fea.CoordinateFrame.globalFrame", "fea.DirectionalCombination.percentage", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`vector` must be a 1-by-3 or 3-by-1 finite real vector with nonzero length; it is ",
                    "normalized to unit length before it is stored, so `[0 0 2]` and `[0 0 1]` are the ",
                    "same direction. `frame` must be a typed `fea.CoordinateFrame` such as ",
                    "`fea.CoordinateFrame.globalFrame()`.",
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
            "vertical",
            "Excite along global z",
            "direction = fea.SpectrumDirection.new(fea.CoordinateFrame.globalFrame(), [0 0 2]);",
            concat!(
                "assert(strcmp(class(direction), 'fea.SpectrumDirection'));\n",
                "assert(direction.value.vector(3) == 1);\n",
                "assert(direction.value.frame == \"global\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Does the vector length scale the spectrum?",
            "No. The direction is normalized; scale belongs in the ordinates.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/spectrum_direction/tests.rs::spectrum_direction_identities_conform",
        ),
        unit_test(
            "Spectrum direction is normalized and portable",
            "crates/runmat-runtime/src/builtins/fea/response_spectrum_policies.rs::direction_is_normalized_and_portable",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPECTRUM_DIRECTION_NEW_CATALOG_ENTRY,
    name: "fea.SpectrumDirection.new",
    result: classes::SPECTRUM_DIRECTION,
    signature: "direction = fea.SpectrumDirection.new(frame, vector)",
    inputs: [
        (
            "frame",
            Any,
            "Typed fea.CoordinateFrame the vector is expressed in.",
            PolicyArg::Typed(crate::fea_contract::model_types::classes::FEA_COORDINATE_FRAME_CLASSES),
        ),
        (
            "vector",
            NumericArray,
            "Finite, nonzero three-component excitation direction.",
            PolicyArg::RealVector3,
        ),
    ],
    output: ("direction", "Typed fea.SpectrumDirection value."),
    integer_capabilities: &integer::SPECTRUM_DIRECTION_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
