use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalDomain.time",
    summary: "Select time in seconds as the domain of a generated signal.",
    description: concat!(
        "`fea.SignalDomain.time()` returns the `fea.SignalDomain` whose `domain` property is ",
        "`time_seconds`: generated signals are sampled in seconds.",
    ),
    keywords: &["fea", "signal", "domain", "time", "generator"],
    related: &["fea.SignalDomain.time", "fea.SignalDomain.frequency", "fea.SignalDomain.studyParameter", "fea.SignalSampling.uniform", "fea.ForceSignal.fromFunction", "fea.PressureSignal.fromFunction", "fea.TemperatureSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Use the time domain for loads and boundary values that vary during a transient ",
                    "procedure. Sample coordinates are seconds and may be negative only if the ",
                    "procedure's time axis starts before zero.",
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
            "time",
            "Select time in seconds as the domain of a generated signal",
            "domain = fea.SignalDomain.time();",
            concat!(
                "assert(strcmp(class(domain), 'fea.SignalDomain'));\n",
                "assert(domain.domain == \"time_seconds\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Are milliseconds supported?",
            "Write the sampling coordinates in seconds; for example 2 ms is 0.002.",
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
    FEA_SIGNAL_DOMAIN_TIME_CATALOG_ENTRY,
    name: "fea.SignalDomain.time",
    result: classes::SIGNAL_DOMAIN,
    signature: "domain = fea.SignalDomain.time()",
    inputs: [
    ],
    output: ("domain", "Typed fea.SignalDomain."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
