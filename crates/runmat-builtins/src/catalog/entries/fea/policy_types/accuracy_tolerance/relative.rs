use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTolerance.relative",
    summary: "Create a dimensionless relative accuracy tolerance.",
    description: concat!(
        "`fea.AccuracyTolerance.relative(value)` returns an `fea.AccuracyTolerance` that ",
        "accepts a result when its estimated error, relative to the magnitude of the assessed",
        " result, is at most `value`. The stored value is the struct `relative: value`.",
    ),
    keywords: &["fea", "accuracy", "tolerance", "relative", "acceptance"],
    related: &["fea.AccuracyTolerance.absolute", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`value` is dimensionless: `0.01` means 1 percent. It accepts double and every ",
                    "built-in integer class; integers are converted once to binary64. Logical, text, ",
                    "complex, and nonscalar inputs fail with `RunMat:fea:InvalidInput`.",
                ),
                concat!(
                    "A relative tolerance is the only form accepted by a criterion whose target is ",
                    "`fea.AccuracyTarget.numericalResult()`.",
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
            "one-percent",
            "Accept a 1 percent relative error",
            "tolerance = fea.AccuracyTolerance.relative(0.01);",
            concat!(
                "assert(strcmp(class(tolerance), 'fea.AccuracyTolerance'));\n",
                "assert(tolerance.value.relative == 0.01);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is 0.01 one percent or 0.01 percent?",
            "One percent. The value is a plain ratio, not a percentage.",
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
    FEA_ACCURACY_TOLERANCE_RELATIVE_CATALOG_ENTRY,
    name: "fea.AccuracyTolerance.relative",
    result: classes::ACCURACY_TOLERANCE,
    signature: "tolerance = fea.AccuracyTolerance.relative(value)",
    inputs: [
        (
            "value",
            NumericScalar,
            concat!(
                "Finite dimensionless bound, for example 0.01 for 1 percent; must be positive when ",
                "the study is validated.",
            ),
            PolicyArg::RealScalar,
        ),
    ],
    output: ("tolerance", "Typed fea.AccuracyTolerance with a relative bound."),
    integer_capabilities: &integer::TOLERANCE_VALUE_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
