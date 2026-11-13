use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationQuantity.electricCurrent",
    summary: "Observe electric current, in amperes.",
    description: concat!(
        "`fea.ObservationQuantity.electricCurrent()` returns the `fea.ObservationQuantity` ",
        "whose `value` is `electric_current`. Electric current is a scalar in A, typically ",
        "the current through a terminal or boundary; it accepts outward- or inward-positive ",
        "sign conventions.",
    ),
    keywords: &["fea", "observation", "observation quantity", "electric current"],
    related: &["fea.ObservationQuantity.displacement", "fea.ObservationQuantity.rotation", "fea.ObservationQuantity.velocity", "fea.ObservationQuantity.acceleration", "fea.Observation.field", "fea.AccuracyTarget.quantity"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Electric current is a scalar in A, typically the current through a terminal or ",
                    "boundary; it accepts outward- or inward-positive sign conventions.",
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
            "electric-current",
            "Observe electric current, in amperes",
            "quantity = fea.ObservationQuantity.electricCurrent();",
            concat!(
                "assert(strcmp(class(quantity), 'fea.ObservationQuantity'));\n",
                "assert(quantity.value == \"electric_current\");",
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
    FEA_OBSERVATION_QUANTITY_ELECTRIC_CURRENT_CATALOG_ENTRY,
    name: "fea.ObservationQuantity.electricCurrent",
    result: classes::OBSERVATION_QUANTITY,
    signature: "quantity = fea.ObservationQuantity.electricCurrent()",
    inputs: [
    ],
    output: ("quantity", "Typed fea.ObservationQuantity value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
