use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationQuantity.displacement",
    summary: "Observe displacement, in meters.",
    description: concat!(
        "`fea.ObservationQuantity.displacement()` returns the `fea.ObservationQuantity` whose",
        " `value` is `displacement`. Displacement is the primary structural solution: a ",
        "three-component vector in meters at nodes, available from static, transient, ",
        "harmonic, and modal steps (as mode shapes).",
    ),
    keywords: &["fea", "observation", "observation quantity", "displacement"],
    related: &["fea.ObservationQuantity.rotation", "fea.ObservationQuantity.velocity", "fea.ObservationQuantity.acceleration", "fea.ObservationQuantity.strain", "fea.Observation.field", "fea.AccuracyTarget.quantity"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Displacement is the primary structural solution: a three-component vector in meters ",
                    "at nodes, available from static, transient, harmonic, and modal steps (as mode ",
                    "shapes).",
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
            "displacement",
            "Observe displacement, in meters",
            "quantity = fea.ObservationQuantity.displacement();",
            concat!(
                "assert(strcmp(class(quantity), 'fea.ObservationQuantity'));\n",
                "assert(quantity.value == \"displacement\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/observation_quantity/tests.rs::observation_quantity_identities_conform",
        ),
        unit_test(
            "Structural recovery quantities preserve typed values",
            "crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs::structural_recovery_quantity_constructors_preserve_every_typed_value",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_OBSERVATION_QUANTITY_DISPLACEMENT_CATALOG_ENTRY,
    name: "fea.ObservationQuantity.displacement",
    result: classes::OBSERVATION_QUANTITY,
    signature: "quantity = fea.ObservationQuantity.displacement()",
    inputs: [
    ],
    output: ("quantity", "Typed fea.ObservationQuantity value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
