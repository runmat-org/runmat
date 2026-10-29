use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalDomain.frequency",
    summary: "Select frequency in hertz as the domain of a generated signal.",
    description: concat!(
        "`fea.SignalDomain.frequency()` returns the `fea.SignalDomain` whose `domain` ",
        "property is `frequency_hertz`: generated signals are sampled in hertz.",
    ),
    keywords: &["fea", "signal", "domain", "frequency", "generator"],
    related: &["fea.SignalDomain.time", "fea.SignalDomain.frequency", "fea.SignalDomain.studyParameter", "fea.SignalSampling.uniform", "fea.ForceSignal.fromFunction", "fea.PressureSignal.fromFunction", "fea.TemperatureSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Use the frequency domain for harmonic and frequency-response loads. Sampling ",
                    "coordinates are in hertz and cannot be negative; a callable signal whose sampling ",
                    "starts below 0 Hz is rejected when it is created and again at study validation.",
                ),
                super::DOMAIN_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::DOMAIN_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "frequency",
            "Select frequency in hertz as the domain of a generated signal",
            "domain = fea.SignalDomain.frequency();",
            concat!(
                "assert(strcmp(class(domain), 'fea.SignalDomain'));\n",
                "assert(domain.domain == \"frequency_hertz\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Are angular frequencies accepted?",
            "No. Coordinates are cyclic frequency in hertz. Convert rad/s by dividing by 2*pi.",
        ),
    ],
    links: &[
        page("RunMat function integration", "/docs/simulation/language/runmat-function-integration"),
        page("Run a transient study with a generated input", "/docs/simulation/language/guides/transient-study-with-generated-inputs"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/generator_policies.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/signal_domain/tests.rs::signal_domain_identities_conform",
        ),
        unit_test(
            "Generator policies are typed, portable, and bounded",
            "crates/runmat-runtime/src/builtins/fea/generator_policies.rs::generator_policies_are_typed_portable_and_validate_bounds",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SIGNAL_DOMAIN_FREQUENCY_CATALOG_ENTRY,
    name: "fea.SignalDomain.frequency",
    result: classes::SIGNAL_DOMAIN,
    signature: "domain = fea.SignalDomain.frequency()",
    inputs: [
    ],
    output: ("domain", "Typed fea.SignalDomain."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
