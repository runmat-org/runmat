use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationComponent.all",
    summary: "Report every component of the quantity.",
    description: concat!(
        "`fea.ObservationComponent.all()` returns the `fea.ObservationComponent` whose ",
        "`value` is `all`. All keeps the quantity's full value shape.",
    ),
    keywords: &["fea", "observation", "observation component", "all"],
    related: &["fea.ObservationComponent.x", "fea.ObservationComponent.y", "fea.ObservationComponent.z", "fea.ObservationComponent.normal", "fea.CoordinateFrame.globalFrame", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "All keeps the quantity's full value shape. It is the only component accepted for ",
                    "scalar quantities and for section-resultant quantities.",
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
            "all",
            "Report every component of the quantity",
            "component = fea.ObservationComponent.all();",
            concat!(
                "assert(strcmp(class(component), 'fea.ObservationComponent'));\n",
                "assert(component.value == \"all\");",
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
    FEA_OBSERVATION_COMPONENT_ALL_CATALOG_ENTRY,
    name: "fea.ObservationComponent.all",
    result: classes::OBSERVATION_COMPONENT,
    signature: "component = fea.ObservationComponent.all()",
    inputs: [
    ],
    output: ("component", "Typed fea.ObservationComponent value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
