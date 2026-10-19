use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalDomain.studyParameter",
    summary: "Select a named study parameter as the domain of a generated signal.",
    description: concat!(
        "`fea.SignalDomain.studyParameter(parameterId)` returns an `fea.SignalDomain` whose ",
        "samples are canonical values of the study parameter `parameterId`, for example a ",
        "load scale swept across studies.",
    ),
    keywords: &["fea", "signal", "domain", "parameter", "sweep"],
    related: &["fea.SignalDomain.time", "fea.SignalSampling.canonicalPoints", "fea.sweep", "fea.ForceSignal.fromFunction", "fea.PressureSignal.fromFunction", "fea.TemperatureSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`parameterId` must be nonempty text; empty or whitespace-only identifiers fail with ",
                    "`RunMat:fea:InvalidInput`. The identifier is stored as written in ",
                    "`domain.study_parameter.parameter_id` and is matched exactly, so sampling a signal ",
                    "requires a sample point for the same parameter.",
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
            "load-scale",
            "Vary a signal with a load-scale parameter",
            "domain = fea.SignalDomain.studyParameter('load_scale');",
            concat!(
                "assert(strcmp(class(domain), 'fea.SignalDomain'));\n",
                "assert(domain.domain.study_parameter.parameter_id == \"load_scale\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is the parameter identifier case sensitive?",
            "Yes. It must match the study parameter's identifier exactly.",
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
    FEA_SIGNAL_DOMAIN_STUDY_PARAMETER_CATALOG_ENTRY,
    name: "fea.SignalDomain.studyParameter",
    result: classes::SIGNAL_DOMAIN,
    signature: "domain = fea.SignalDomain.studyParameter(parameterId)",
    inputs: [
        (
            "parameterId",
            StringScalar,
            "Nonempty identifier of the study parameter the signal varies with.",
            PolicyArg::Text,
        ),
    ],
    output: ("domain", "Typed fea.SignalDomain."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
