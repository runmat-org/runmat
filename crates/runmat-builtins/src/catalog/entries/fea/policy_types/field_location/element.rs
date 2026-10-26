use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.FieldLocation.element",
    summary: "Observe one value per element.",
    description: concat!(
        "`fea.FieldLocation.element()` returns the `fea.FieldLocation` whose `value` is ",
        "`element`. Element is a per-element location, for example an element-averaged ",
        "stress.",
    ),
    keywords: &["fea", "observation", "field location", "element"],
    related: &["fea.FieldLocation.node", "fea.FieldLocation.point", "fea.FieldLocation.cell", "fea.FieldLocation.integrationPoint", "fea.RecoveryPolicy.native", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                "Element is a per-element location, for example an element-averaged stress.",
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
            "element",
            "Observe one value per element",
            "location = fea.FieldLocation.element();",
            concat!(
                "assert(strcmp(class(location), 'fea.FieldLocation'));\n",
                "assert(location.value == \"element\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/field_location/tests.rs::field_location_identities_conform",
        ),
        unit_test(
            "Point field location is preserved",
            "crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs::point_field_location_constructor_preserves_the_typed_location",
        ),
        unit_test(
            "Constraint field location is preserved",
            "crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs::constraint_field_location_constructor_preserves_the_typed_location",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_FIELD_LOCATION_ELEMENT_CATALOG_ENTRY,
    name: "fea.FieldLocation.element",
    result: classes::FIELD_LOCATION,
    signature: "location = fea.FieldLocation.element()",
    inputs: [
    ],
    output: ("location", "Typed fea.FieldLocation value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
