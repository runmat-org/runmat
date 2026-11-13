use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationComponent.y",
    summary: "Report the y component of a vector quantity.",
    description: concat!(
        "`fea.ObservationComponent.y()` returns the `fea.ObservationComponent` whose `value` ",
        "is `y`. Y selects the second component of a vector quantity in the observation's ",
        "coordinate frame.",
    ),
    keywords: &["fea", "observation", "observation component", "y"],
    related: &["fea.ObservationComponent.all", "fea.ObservationComponent.x", "fea.ObservationComponent.z", "fea.ObservationComponent.normal", "fea.CoordinateFrame.globalFrame", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Y selects the second component of a vector quantity in the observation's coordinate ",
                    "frame.",
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
            "y",
            "Report the y component of a vector quantity",
            "component = fea.ObservationComponent.y();",
            concat!(
                "assert(strcmp(class(component), 'fea.ObservationComponent'));\n",
                "assert(component.value == \"y\");",
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
    FEA_OBSERVATION_COMPONENT_Y_CATALOG_ENTRY,
    name: "fea.ObservationComponent.y",
    result: classes::OBSERVATION_COMPONENT,
    signature: "component = fea.ObservationComponent.y()",
    inputs: [
    ],
    output: ("component", "Typed fea.ObservationComponent value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
