use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTarget.quantity",
    summary: "Target a physical quantity of a procedure step for accuracy acceptance.",
    description: concat!(
        "`fea.AccuracyTarget.quantity(quantity)` returns an `fea.AccuracyTarget` that ",
        "assesses one physical quantity, such as displacement or von Mises stress, across the",
        " step rather than one named observation.",
    ),
    keywords: &["fea", "accuracy", "target", "quantity", "acceptance"],
    related: &["fea.ObservationQuantity.displacement", "fea.ObservationQuantity.vonMisesStress", "fea.AccuracyTarget.observation", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`quantity` must be a typed `fea.ObservationQuantity` value; text such as `'stress'` ",
                    "is rejected with `RunMat:fea:InvalidInput`. The quantity is stored in the target's ",
                    "`value` as its canonical snake_case spelling.",
                ),
                concat!(
                    "At study validation the quantity must belong to the step's physics family; for ",
                    "example, a thermal step cannot accept a criterion on von Mises stress. An absolute ",
                    "tolerance on the same criterion must name the same quantity.",
                ),
                super::TARGET_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Use a quantity target when the criterion should hold for a quantity wherever the ",
                    "step reports it, independent of any one observation request.",
                ),
                super::TARGET_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "von-mises",
            "Bound the von Mises stress error to 5 MPa",
            concat!(
                "q = fea.ObservationQuantity.vonMisesStress();\n",
                "target = fea.AccuracyTarget.quantity(q);\n",
                "tolerance = fea.AccuracyTolerance.absolute(q, 5e6);",
            ),
            concat!(
                "assert(strcmp(class(target), 'fea.AccuracyTarget'));\n",
                "assert(target.value.quantity == \"von_mises_stress\");\n",
                "assert(tolerance.value.absolute.canonical_value == 5e6);",
            ),
        ),
    ],
    faqs: &[
        super::TARGET_FAQ,
    ],
    links: &[
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/accuracy.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_target/tests.rs::accuracy_target_identities_conform",
        ),
        unit_test(
            "Targets preserve typed values and reject untyped inputs",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::targets_preserve_typed_values_and_reject_empty_or_untyped_inputs",
        ),
        unit_test(
            "Accuracy plan validation rejects nonpositive tolerances",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_nonpositive_tolerance",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_TARGET_QUANTITY_CATALOG_ENTRY,
    name: "fea.AccuracyTarget.quantity",
    result: classes::ACCURACY_TARGET,
    signature: "target = fea.AccuracyTarget.quantity(quantity)",
    inputs: [
        (
            "quantity",
            Any,
            "Typed fea.ObservationQuantity, for example fea.ObservationQuantity.vonMisesStress().",
            PolicyArg::Typed(&[classes::OBSERVATION_QUANTITY]),
        ),
    ],
    output: ("target", "Typed fea.AccuracyTarget naming one physical quantity."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
