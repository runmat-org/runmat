use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.FieldLocation.referencePoint",
    summary: "Observe values at reference points.",
    description: concat!(
        "`fea.FieldLocation.referencePoint()` returns the `fea.FieldLocation` whose `value` ",
        "is `reference_point`. Reference point is the location of reference-point degrees of ",
        "freedom, such as the rigid-body motion of a coupled reference node.",
    ),
    keywords: &["fea", "observation", "field location", "reference point"],
    related: &["fea.FieldLocation.node", "fea.FieldLocation.point", "fea.FieldLocation.cell", "fea.FieldLocation.element", "fea.RecoveryPolicy.native", "fea.Observation.field"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Reference point is the location of reference-point degrees of freedom, such as the ",
                    "rigid-body motion of a coupled reference node.",
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
            "reference-point",
            "Observe values at reference points",
            "location = fea.FieldLocation.referencePoint();",
            concat!(
                "assert(strcmp(class(location), 'fea.FieldLocation'));\n",
                "assert(location.value == \"reference_point\");",
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
    FEA_FIELD_LOCATION_REFERENCE_POINT_CATALOG_ENTRY,
    name: "fea.FieldLocation.referencePoint",
    result: classes::FIELD_LOCATION,
    signature: "location = fea.FieldLocation.referencePoint()",
    inputs: [
    ],
    output: ("location", "Typed fea.FieldLocation value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
