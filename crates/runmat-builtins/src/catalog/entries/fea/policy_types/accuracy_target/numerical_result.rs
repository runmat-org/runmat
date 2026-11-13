use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTarget.numericalResult",
    summary: "Target the complete numerical result of a step for accuracy acceptance.",
    description: concat!(
        "`fea.AccuracyTarget.numericalResult()` returns the `fea.AccuracyTarget` whose ",
        "`value` is `numerical_result`. It assesses whole-result measures such as modal mass ",
        "capture or frequency-quadrature convergence.",
    ),
    keywords: &["fea", "accuracy", "target", "numerical result", "acceptance"],
    related: &["fea.AccuracyTarget.observation", "fea.AccuracyTarget.quantity", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "A numerical-result criterion must use a dimensionless ",
                    "`fea.AccuracyTolerance.relative` tolerance; an absolute tolerance is rejected at ",
                    "study validation. Its error contribution may be any kind except observation sampling",
                    " and execution equivalence. RunMat's built-in modal response-spectrum and ",
                    "random-response steps use this target for modal-truncation and frequency-sampling ",
                    "evidence.",
                ),
                super::TARGET_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Use this target when the acceptance measure belongs to the solution as a whole ",
                    "rather than to one physical quantity, for example 90 percent cumulative modal mass ",
                    "in a response-spectrum step.",
                ),
                super::TARGET_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "numerical-result",
            "Select the numerical result target",
            "target = fea.AccuracyTarget.numericalResult();",
            concat!(
                "assert(strcmp(class(target), 'fea.AccuracyTarget'));\n",
                "assert(target.value == \"numerical_result\");",
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
    FEA_ACCURACY_TARGET_NUMERICAL_RESULT_CATALOG_ENTRY,
    name: "fea.AccuracyTarget.numericalResult",
    result: classes::ACCURACY_TARGET,
    signature: "target = fea.AccuracyTarget.numericalResult()",
    inputs: [
    ],
    output: ("target", "Typed fea.AccuracyTarget value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
