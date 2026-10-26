use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationPurpose.visualization",
    summary: "Declare an observation for visualization.",
    description: concat!(
        "`fea.ObservationPurpose.visualization()` returns the `fea.ObservationPurpose` whose ",
        "`value` is `visualization`. Visualization marks a result intended for plots and ",
        "field views.",
    ),
    keywords: &["fea", "observation", "observation purpose", "visualization"],
    related: &["fea.ObservationPurpose.publication", "fea.ObservationPurpose.solverControl", "fea.ObservationPurpose.downstreamState", "fea.Observation.field", "fea.CompressionPolicy.errorBounded"],
    sections: &[
        section(
            "Behavior",
            &[
                "Visualization marks a result intended for plots and field views.",
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
            "visualization",
            "Declare an observation for visualization",
            "purpose = fea.ObservationPurpose.visualization();",
            concat!(
                "assert(strcmp(class(purpose), 'fea.ObservationPurpose'));\n",
                "assert(purpose.value == \"visualization\");",
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
    FEA_OBSERVATION_PURPOSE_VISUALIZATION_CATALOG_ENTRY,
    name: "fea.ObservationPurpose.visualization",
    result: classes::OBSERVATION_PURPOSE,
    signature: "purpose = fea.ObservationPurpose.visualization()",
    inputs: [
    ],
    output: ("purpose", "Typed fea.ObservationPurpose value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
