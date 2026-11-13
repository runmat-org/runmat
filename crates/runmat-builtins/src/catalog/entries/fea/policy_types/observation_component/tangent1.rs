use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationComponent.tangent1",
    summary: "Report the first boundary-tangent component.",
    description: concat!(
        "`fea.ObservationComponent.tangent1()` returns the `fea.ObservationComponent` whose ",
        "`value` is `tangent1`. Tangent1 selects the component along the first boundary ",
        "tangent of a vector or symmetric-tensor quantity.",
    ),
    keywords: &["fea", "observation", "observation component", "tangent1"],
    related: &["fea.ObservationComponent.all", "fea.ObservationComponent.x", "fea.ObservationComponent.y", "fea.ObservationComponent.z", "fea.CoordinateFrame.globalFrame", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Tangent1 selects the component along the first boundary tangent of a vector or ",
                    "symmetric-tensor quantity. It requires the boundary-normal/tangent frame.",
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
            "tangent1",
            "Report the first boundary-tangent component",
            "component = fea.ObservationComponent.tangent1();",
            concat!(
                "assert(strcmp(class(component), 'fea.ObservationComponent'));\n",
                "assert(component.value == \"tangent1\");",
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
    FEA_OBSERVATION_COMPONENT_TANGENT1_CATALOG_ENTRY,
    name: "fea.ObservationComponent.tangent1",
    result: classes::OBSERVATION_COMPONENT,
    signature: "component = fea.ObservationComponent.tangent1()",
    inputs: [
    ],
    output: ("component", "Typed fea.ObservationComponent value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
