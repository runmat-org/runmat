use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyPlan.baseline",
    summary: "Create the baseline accuracy plan that step constructors attach by default.",
    description: concat!(
        "`fea.AccuracyPlan.baseline(id)` returns the baseline `fea.AccuracyPlan`: a mandatory ",
        "algebraic-residual bound that every procedure step carries until a study authors its own plan.",
    ),
    keywords: &["fea", "accuracy", "plan", "baseline", "residual"],
    related: &["fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The plan has exactly one mandatory criterion, `algebraic_residual`: target ",
                    "`globalEquilibrium`, contribution `algebraic`, relative tolerance 1e-8, evidence ",
                    "`residualBound`. Unavailable evidence is marked indeterminate, the ",
                    "limit action is `fail`, and numerical equivalence is a relative tolerance of 1e-10.",
                ),
                concat!(
                    "Its resource ceiling allows no automatic refinement or solver tightening, 1,000,000 ",
                    "samples, 3,600,000 ms (one hour) of wall time, and 8 GiB (8,589,934,592 bytes) of ",
                    "resident memory.",
                ),
                "`id` must be nonempty text. No other argument is accepted.",
            ],
        ),
        section(
            "Use in a study",
            &[
                super::PLAN_STUDY,
                concat!(
                    "The baseline checks algebraic equilibrium only. Author a plan with ",
                    "`fea.AccuracyPlan.new` to add discretization, sampling, or reference-comparison ",
                    "criteria for a study's acceptance.",
                ),
            ],
        ),
    ],
    examples: &[
        example(
            "baseline",
            "Inspect the baseline plan",
            concat!(
                "plan = fea.AccuracyPlan.baseline('bracket_static.accuracy');\n",
                "criterion = plan.criteria{1};",
            ),
            concat!(
                "assert(strcmp(class(plan), 'fea.AccuracyPlan'));\n",
                "assert(plan.accuracy_plan_id == \"bracket_static.accuracy\");\n",
                "assert(plan.unavailable_evidence == \"fail\");\n",
                "assert(plan.resources.maximum_refinement_cycles == 0);\n",
                "assert(criterion.criterion_id == \"algebraic_residual\");",
            ),
        ),
    ],
    faqs: &[
        super::PLAN_FAQ,
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_plan/tests.rs::accuracy_plan_identities_conform",
        ),
        unit_test(
            "Baseline matches the core baseline",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::baseline_matches_the_core_baseline",
        ),
        unit_test(
            "Baseline validates",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::baseline_is_valid",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_PLAN_BASELINE_CATALOG_ENTRY,
    name: "fea.AccuracyPlan.baseline",
    result: classes::ACCURACY_PLAN,
    signature: "plan = fea.AccuracyPlan.baseline(id)",
    inputs: [
        (
            "id",
            StringScalar,
            "Nonempty accuracy-plan identifier.",
            PolicyArg::Text,
        ),
    ],
    output: ("plan", "Typed fea.AccuracyPlan with the baseline contents."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
