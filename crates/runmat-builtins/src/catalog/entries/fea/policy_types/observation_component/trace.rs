use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationComponent.trace",
    summary: "Report the trace of a tensor quantity.",
    description: concat!(
        "`fea.ObservationComponent.trace()` returns the `fea.ObservationComponent` whose ",
        "`value` is `trace`. Trace is the sum of the diagonal components of a symmetric ",
        "tensor such as stress or strain.",
    ),
    keywords: &["fea", "observation", "observation component", "trace"],
    related: &["fea.ObservationComponent.all", "fea.ObservationComponent.x", "fea.ObservationComponent.y", "fea.ObservationComponent.z", "fea.CoordinateFrame.globalFrame", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Trace is the sum of the diagonal components of a symmetric tensor such as stress or ",
                    "strain.",
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
            "trace",
            "Report the trace of a tensor quantity",
            "component = fea.ObservationComponent.trace();",
            concat!(
                "assert(strcmp(class(component), 'fea.ObservationComponent'));\n",
                "assert(component.value == \"trace\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/observation_component/tests.rs::observation_component_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_OBSERVATION_COMPONENT_TRACE_CATALOG_ENTRY,
    name: "fea.ObservationComponent.trace",
    result: classes::OBSERVATION_COMPONENT,
    signature: "component = fea.ObservationComponent.trace()",
    inputs: [
    ],
    output: ("component", "Typed fea.ObservationComponent value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
