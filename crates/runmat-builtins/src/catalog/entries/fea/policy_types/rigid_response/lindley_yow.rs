use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RigidResponse.lindleyYow",
    summary: concat!(
        "Separate periodic and rigid modal response with the Lindley-Yow method at an ",
        "explicit zero-period acceleration.",
    ),
    description: concat!(
        "`fea.RigidResponse.lindleyYow(zeroPeriodAcceleration)` returns an ",
        "`fea.RigidResponse` that splits each mode's response into periodic and rigid parts ",
        "from the ratio of the zero-period acceleration to the spectral acceleration at that ",
        "mode.",
    ),
    keywords: &["fea", "response spectrum", "rigid response", "lindley yow"],
    related: &["fea.RigidResponse.gupta", "fea.RigidResponse.disabled", "fea.ModalCombination.completeQuadratic", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The ZPA must be a finite, nonnegative real scalar in m/s^2; a negative value fails ",
                    "with `RunMat:fea:InvalidInput`. The stored value is ",
                    "`value.lindley_yow.zero_period_acceleration`.",
                ),
                concat!(
                    "When the spectrum is created, rigid-response separation requires an acceleration ",
                    "spectrum and CQC modal combination.",
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
            "lindley-yow",
            "Use Lindley-Yow separation with a 0.5 g ZPA",
            "policy = fea.RigidResponse.lindleyYow(4.9);",
            concat!(
                "assert(strcmp(class(policy), 'fea.RigidResponse'));\n",
                "assert(policy.value.lindley_yow.zero_period_acceleration == 4.9);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is the ZPA in g?",
            "No. It is in m/s^2.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/rigid_response/tests.rs::rigid_response_identities_conform",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_RIGID_RESPONSE_LINDLEY_YOW_CATALOG_ENTRY,
    name: "fea.RigidResponse.lindleyYow",
    result: classes::RIGID_RESPONSE,
    signature: "policy = fea.RigidResponse.lindleyYow(zeroPeriodAcceleration)",
    inputs: [
        (
            "zeroPeriodAcceleration",
            NumericScalar,
            "Nonnegative zero-period acceleration in m/s^2.",
            PolicyArg::RealScalar,
        ),
    ],
    output: ("policy", "Typed fea.RigidResponse value."),
    integer_capabilities: &integer::SPECTRUM_SCALAR_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
