use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationPurpose.downstreamState",
    summary: "Declare an observation that feeds downstream procedure state.",
    description: concat!(
        "`fea.ObservationPurpose.downstreamState()` returns the `fea.ObservationPurpose` ",
        "whose `value` is `downstream_state`. Downstream state marks a result that later ",
        "procedure steps consume.",
    ),
    keywords: &["fea", "observation", "observation purpose", "downstream state"],
    related: &["fea.ObservationPurpose.publication", "fea.ObservationPurpose.solverControl", "fea.ObservationPurpose.visualization", "fea.Observation.field", "fea.CompressionPolicy.errorBounded"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Downstream state marks a result that later procedure steps consume. It cannot use ",
                    "error-bounded (lossy) compression.",
                ),
                super::ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "downstream-state",
            "Declare an observation that feeds downstream procedure state",
            "purpose = fea.ObservationPurpose.downstreamState();",
            concat!(
                "assert(strcmp(class(purpose), 'fea.ObservationPurpose'));\n",
                "assert(purpose.value == \"downstream_state\");",
            ),
        ),
    ],
    faqs: &[
        super::FAQ,
    ],
    links: &[
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
        page("Outputs, fields, and observations", "/docs/simulation/language/outputs-fields-and-observations"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/observation_purpose/tests.rs::observation_purpose_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_OBSERVATION_PURPOSE_DOWNSTREAM_STATE_CATALOG_ENTRY,
    name: "fea.ObservationPurpose.downstreamState",
    result: classes::OBSERVATION_PURPOSE,
    signature: "purpose = fea.ObservationPurpose.downstreamState()",
    inputs: [
    ],
    output: ("purpose", "Typed fea.ObservationPurpose value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
