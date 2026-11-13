use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ObservationQuantity.beamTorsionStress",
    summary: "Observe beam torsional shear stress, in pascals.",
    description: concat!(
        "`fea.ObservationQuantity.beamTorsionStress()` returns the `fea.ObservationQuantity` ",
        "whose `value` is `beam_torsion_stress`. Beam torsion stress is the scalar torsional ",
        "shear stress in Pa at the section's critical location.",
    ),
    keywords: &["fea", "observation", "observation quantity", "beam torsion stress"],
    related: &["fea.ObservationQuantity.displacement", "fea.ObservationQuantity.rotation", "fea.ObservationQuantity.velocity", "fea.ObservationQuantity.acceleration", "fea.Observation.field", "fea.AccuracyTarget.quantity"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Beam torsion stress is the scalar torsional shear stress in Pa at the section's ",
                    "critical location.",
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
            "beam-torsion-stress",
            "Observe beam torsional shear stress, in pascals",
            "quantity = fea.ObservationQuantity.beamTorsionStress();",
            concat!(
                "assert(strcmp(class(quantity), 'fea.ObservationQuantity'));\n",
                "assert(quantity.value == \"beam_torsion_stress\");",
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
    FEA_OBSERVATION_QUANTITY_BEAM_TORSION_STRESS_CATALOG_ENTRY,
    name: "fea.ObservationQuantity.beamTorsionStress",
    result: classes::OBSERVATION_QUANTITY,
    signature: "quantity = fea.ObservationQuantity.beamTorsionStress()",
    inputs: [
    ],
    output: ("quantity", "Typed fea.ObservationQuantity value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
