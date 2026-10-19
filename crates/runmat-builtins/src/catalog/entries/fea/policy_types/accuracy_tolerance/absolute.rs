use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTolerance.absolute",
    summary: "Create a dimensioned absolute accuracy tolerance in canonical SI units.",
    description: concat!(
        "`fea.AccuracyTolerance.absolute(quantity, canonicalValue)` returns an ",
        "`fea.AccuracyTolerance` that accepts a result when its estimated error is at most ",
        "`canonicalValue`, expressed in the canonical SI unit of `quantity`.",
    ),
    keywords: &["fea", "accuracy", "tolerance", "absolute", "SI units"],
    related: &["fea.AccuracyTolerance.relative", "fea.ObservationQuantity.displacement", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The bound carries the unit of its quantity: meters for displacement and contact gap,",
                    " pascals for stress and pressure, kelvin for temperature, newtons for reaction ",
                    "force, and so on. There is no unit conversion argument; write the value in SI. ",
                    "`canonicalValue` accepts double and every built-in integer class.",
                ),
                concat!(
                    "At study validation the tolerance quantity must equal the quantity of the ",
                    "criterion's target: the quantity named by `fea.AccuracyTarget.quantity`, or the ",
                    "quantity of the observation named by `fea.AccuracyTarget.observation`. A ",
                    "whole-result target (`numericalResult`) cannot use an absolute tolerance.",
                ),
                super::TOLERANCE_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::TOLERANCE_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "tip-displacement-0-1-mm",
            "Bound a displacement error to 0.1 mm",
            concat!(
                "q = fea.ObservationQuantity.displacement();\n",
                "tolerance = fea.AccuracyTolerance.absolute(q, 1e-4);",
            ),
            concat!(
                "assert(strcmp(class(tolerance), 'fea.AccuracyTolerance'));\n",
                "assert(tolerance.value.absolute.quantity == \"displacement\");\n",
                "assert(tolerance.value.absolute.canonical_value == 1e-4);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "How do I write 0.1 mm?",
            concat!(
                "Write `1e-4`. Displacement is stored in meters, so absolute displacement tolerances ",
                "are always meters.",
            ),
        ),
        faq(
            "Can I use an absolute tolerance with an observation target?",
            concat!(
                "Yes, if the tolerance quantity matches the observation's quantity. A mismatch fails ",
                "study validation.",
            ),
        ),
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_tolerance/tests.rs::accuracy_tolerance_identities_conform",
        ),
        unit_test(
            "Tolerances store finite scalars",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::tolerances_and_equivalence_store_finite_scalars_and_reject_non_numeric_inputs",
        ),
        unit_test(
            "Accuracy plan validation rejects nonpositive tolerances",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_nonpositive_tolerance",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_TOLERANCE_ABSOLUTE_CATALOG_ENTRY,
    name: "fea.AccuracyTolerance.absolute",
    result: classes::ACCURACY_TOLERANCE,
    signature: "tolerance = fea.AccuracyTolerance.absolute(quantity, canonicalValue)",
    inputs: [
        (
            "quantity",
            Any,
            "Typed fea.ObservationQuantity whose canonical SI unit gives the bound its unit.",
            PolicyArg::Typed(&[classes::OBSERVATION_QUANTITY]),
        ),
        (
            "canonicalValue",
            NumericScalar,
            concat!(
                "Finite bound in the quantity's canonical SI unit (for example meters or pascals); ",
                "must be positive when the study is validated.",
            ),
            PolicyArg::RealScalar,
        ),
    ],
    output: ("tolerance", "Typed fea.AccuracyTolerance with a dimensioned bound."),
    integer_capabilities: &integer::ABSOLUTE_TOLERANCE_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
