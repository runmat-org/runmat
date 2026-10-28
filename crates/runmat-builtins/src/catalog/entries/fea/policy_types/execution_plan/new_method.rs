use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ExecutionPlan.new",
    summary: "Create an execution plan from explicit per-step execution bindings.",
    description: concat!(
        "`fea.ExecutionPlan.new(sequence, steps)` returns an `fea.ExecutionPlan` whose ",
        "`steps` property holds the supplied execution steps after checking them against ",
        "`sequence`.",
    ),
    keywords: &["fea", "execution", "plan", "sequence", "solver"],
    related: &["fea.ExecutionStep.new", "fea.ExecutionPlan.defaults", "fea.AnalysisSequence.new", "fea.study"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The plan must contain exactly one execution step for each procedure step of the ",
                    "sequence, with no duplicates and no identifiers absent from the sequence, and each ",
                    "step's solver options must support that procedure. Any mismatch fails with ",
                    "`RunMat:fea:InvalidInput` before a plan is returned.",
                ),
                concat!(
                    "The sequence is used only for this check; the returned plan stores the execution ",
                    "steps, not a copy of the sequence.",
                ),
                super::EXECUTION_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::EXECUTION_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "explicit-plan",
            "Build an explicit plan for a one-step sequence",
            concat!(
                "static = fea.StructuralStep.static('preload');\n",
                "sequence = fea.AnalysisSequence.new('bracket', {static}, {}, {}, {});\n",
                "execution = fea.ExecutionStep.new(static, fea.ComputeBackend.cpu(), ...\n",
                "    fea.SolverOptions.defaults(static));\n",
                "plan = fea.ExecutionPlan.new(sequence, {execution});",
            ),
            concat!(
                "assert(strcmp(class(plan), 'fea.ExecutionPlan'));\n",
                "assert(numel(plan.steps) == 1);\n",
                "assert(plan.steps{1}.step_id == \"preload\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Must the steps be in sequence order?",
            concat!(
                "No. Each entry is matched to its procedure step by identifier; the sequence alone ",
                "defines execution order and state flow.",
            ),
        ),
    ],
    links: &[
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/execution.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/execution_plan/tests.rs::execution_plan_identities_conform",
        ),
        unit_test(
            "Explicit plans must cover the whole sequence",
            "crates/runmat-runtime/src/builtins/fea/execution.rs::execution_plan_rejects_missing_sequence_steps",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_EXECUTION_PLAN_NEW_CATALOG_ENTRY,
    name: "fea.ExecutionPlan.new",
    result: classes::EXECUTION_PLAN,
    signature: "plan = fea.ExecutionPlan.new(sequence, steps)",
    inputs: [
        (
            "sequence",
            Any,
            "Typed fea.AnalysisSequence the plan executes.",
            PolicyArg::Typed(&[classes::ANALYSIS_SEQUENCE]),
        ),
        (
            "steps",
            Any,
            "One fea.ExecutionStep or a cell array with exactly one entry per sequence step.",
            PolicyArg::TypedOrCell(&[classes::EXECUTION_STEP]),
        ),
    ],
    output: ("plan", "Typed fea.ExecutionPlan."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
