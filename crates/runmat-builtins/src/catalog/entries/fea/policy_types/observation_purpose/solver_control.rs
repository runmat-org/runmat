use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationPurpose.solverControl",
    summary: "Declare an observation that feeds solver control.",
    description: concat!(
        "`fea.ObservationPurpose.solverControl()` returns the `fea.ObservationPurpose` whose ",
        "`value` is `solver_control`. Solver control marks a result that drives the solve, ",
        "such as a monitored residual.",
    ),
    keywords: &["fea", "observation", "observation purpose", "solver control"],
    related: &["fea.ObservationPurpose.publication", "fea.ObservationPurpose.visualization", "fea.ObservationPurpose.downstreamState", "fea.Observation.field", "fea.CompressionPolicy.errorBounded"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Solver control marks a result that drives the solve, such as a monitored residual. ",
                    "It cannot use error-bounded (lossy) compression.",
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
            "solver-control",
            "Declare an observation that feeds solver control",
            "purpose = fea.ObservationPurpose.solverControl();",
            concat!(
                "assert(strcmp(class(purpose), 'fea.ObservationPurpose'));\n",
                "assert(purpose.value == \"solver_control\");",
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
    FEA_OBSERVATION_PURPOSE_SOLVER_CONTROL_CATALOG_ENTRY,
    name: "fea.ObservationPurpose.solverControl",
    result: classes::OBSERVATION_PURPOSE,
    signature: "purpose = fea.ObservationPurpose.solverControl()",
    inputs: [
    ],
    output: ("purpose", "Typed fea.ObservationPurpose value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
