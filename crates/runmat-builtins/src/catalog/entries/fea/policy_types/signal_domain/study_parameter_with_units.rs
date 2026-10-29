use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalDomain.studyParameterWithUnits",
    summary: "Select a named study parameter with explicit canonical units as the domain of a generated signal.",
    description: concat!(
        "`fea.SignalDomain.studyParameterWithUnits(parameterId, canonicalUnits)` returns an ",
        "`fea.SignalDomain` whose samples are canonical values of the study parameter ",
        "`parameterId`, and records the physical quantity of that axis from `canonicalUnits`.",
    ),
    keywords: &["fea", "signal", "domain", "parameter", "units", "sweep"],
    related: &["fea.SignalDomain.studyParameter", "fea.SignalSampling.canonicalPoints", "fea.sweep", "fea.ForceSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`parameterId` must be nonempty text. `canonicalUnits` must parse as a unit ",
                    "expression with scale 1 and no offset (for example `m`, `N`, `rad`, `K` or `1`): ",
                    "sampling-policy numbers are already canonical, so a scaled unit such as `mm` fails ",
                    "with `RunMat:fea:InvalidInput`. The domain stores the unit's physical dimension.",
                ),
                super::DOMAIN_ROLE,
            ],
        ),
        section("Use in a study", &[super::DOMAIN_STUDY]),
    ],
    examples: &[example(
        "gap-length",
        "Vary a signal with a length-valued parameter",
        "domain = fea.SignalDomain.studyParameterWithUnits('gap', 'm');",
        "assert(strcmp(class(domain), 'fea.SignalDomain'));",
    )],
    faqs: &[faq(
        "Can I pass millimeters?",
        "No. Give the canonical unit (m) and convert the sample coordinates to it first.",
    )],
    links: &[page("RunMat function integration", "/docs/simulation/language/runmat-function-integration")],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/generator_policies.rs",
    )],
    verification: &[unit_test(
        "Catalog contract conformance",
        "crates/runmat-builtins/src/catalog/entries/fea/policy_types/signal_domain/tests.rs::signal_domain_identities_conform",
    )],
    notes: &[],
});

policy_entry!(
    FEA_SIGNAL_DOMAIN_STUDY_PARAMETER_WITH_UNITS_CATALOG_ENTRY,
    name: "fea.SignalDomain.studyParameterWithUnits",
    result: classes::SIGNAL_DOMAIN,
    signature: "domain = fea.SignalDomain.studyParameterWithUnits(parameterId, canonicalUnits)",
    inputs: [
        (
            "parameterId",
            StringScalar,
            "Nonempty identifier of the study parameter the signal varies with.",
            PolicyArg::Text,
        ),
        (
            "canonicalUnits",
            StringScalar,
            "Canonical coordinate units (m, N, rad, 1, etc.); sampling policy numbers are already canonical.",
            PolicyArg::Text,
        ),
    ],
    output: ("domain", "Typed fea.SignalDomain."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
