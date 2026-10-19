use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RigidResponse.gupta",
    summary: "Separate periodic and rigid modal response with the Gupta method.",
    description: concat!(
        "`fea.RigidResponse.gupta(periodicFrequencyHertz, rigidFrequencyHertz)` returns an ",
        "`fea.RigidResponse` that treats modes below the periodic frequency as periodic, ",
        "modes above the rigid frequency as rigid, and blends linearly in between.",
    ),
    keywords: &["fea", "response spectrum", "rigid response", "gupta"],
    related: &["fea.RigidResponse.lindleyYow", "fea.RigidResponse.disabled", "fea.ModalCombination.completeQuadratic", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Both frequencies are finite and in hertz, with `0 < periodicFrequencyHertz < ",
                    "rigidFrequencyHertz`; other values fail with `RunMat:fea:InvalidInput`. The stored ",
                    "value is `value.gupta` with `periodic_frequency_hz` and `rigid_frequency_hz`.",
                ),
                concat!(
                    "When the spectrum is created, any rigid-response separation requires an acceleration",
                    " spectrum and `fea.ModalCombination.completeQuadratic()` modal combination.",
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
            "gupta-8-33",
            "Use Gupta separation between 8 and 33 Hz",
            "policy = fea.RigidResponse.gupta(8, 33);",
            concat!(
                "assert(strcmp(class(policy), 'fea.RigidResponse'));\n",
                "assert(policy.value.gupta.rigid_frequency_hz == 33);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Why does my spectrum reject Gupta with SRSS?",
            concat!(
                "Rigid-response separation is defined with CQC modal combination. Use ",
                "`fea.ModalCombination.completeQuadratic()`.",
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
    FEA_RIGID_RESPONSE_GUPTA_CATALOG_ENTRY,
    name: "fea.RigidResponse.gupta",
    result: classes::RIGID_RESPONSE,
    signature: "policy = fea.RigidResponse.gupta(periodicFrequencyHertz, rigidFrequencyHertz)",
    inputs: [
        (
            "periodicFrequencyHertz",
            NumericScalar,
            "Upper frequency of the wholly periodic region, in hertz.",
            PolicyArg::RealScalar,
        ),
        (
            "rigidFrequencyHertz",
            NumericScalar,
            concat!(
                "Lower frequency of the wholly rigid region, in hertz; greater than the periodic ",
                "frequency.",
            ),
            PolicyArg::RealScalar,
        ),
    ],
    output: ("policy", "Typed fea.RigidResponse value."),
    integer_capabilities: &integer::GUPTA_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
