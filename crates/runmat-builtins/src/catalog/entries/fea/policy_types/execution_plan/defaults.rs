use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ExecutionPlan.defaults",
    summary: "Create an execution plan with default solver options for every step of a sequence.",
    description: concat!(
        "`fea.ExecutionPlan.defaults(sequence, backend)` returns an ",
        "`fea.ExecutionPlan` with one execution step per procedure step, each using ",
        "`backend` and `fea.SolverOptions.defaults` for that step.",
    ),
    keywords: &["fea", "execution", "plan", "defaults", "backend"],
    related: &["fea.ExecutionPlan.new", "fea.SolverOptions.defaults", "fea.ComputeBackend.cpu", "fea.study"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Default solver options are selected for each step in sequence order. If any step has",
                    " no default (for example a static electromagnetic step), the call fails with ",
                    "`RunMat:fea:InvalidInput` and no plan is returned; build that step with ",
                    "`fea.ExecutionStep.new` and use `fea.ExecutionPlan.new` instead.",
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
            "cpu",
            "Run every step on the CPU",
            concat!(
                "static = fea.StructuralStep.static('preload');\n",
                "sequence = fea.AnalysisSequence.new('bracket', {static}, {}, {}, {});\n",
                "plan = fea.ExecutionPlan.defaults(sequence, fea.ComputeBackend.cpu());",
            ),
            concat!(
                "assert(strcmp(class(plan), 'fea.ExecutionPlan'));\n",
                "assert(numel(plan.steps) == 1);\n",
                "assert(plan.steps{1}.backend == \"cpu\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is this what fea.study does without an Execution option?",
            "Yes. `fea.study` builds the same default plan with the CPU backend.",
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
            "Default plans cover every sequence step",
            "crates/runmat-runtime/src/builtins/fea/execution.rs::typed_execution_plan_covers_each_sequence_step",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_EXECUTION_PLAN_DEFAULTS_CATALOG_ENTRY,
    name: "fea.ExecutionPlan.defaults",
    result: classes::EXECUTION_PLAN,
    signature: "plan = fea.ExecutionPlan.defaults(sequence, backend)",
    inputs: [
        (
            "sequence",
            Any,
            "Typed fea.AnalysisSequence.",
            PolicyArg::Typed(&[classes::ANALYSIS_SEQUENCE]),
        ),
        (
            "backend",
            Any,
            "Typed fea.ComputeBackend applied to every step.",
            PolicyArg::Typed(&[classes::COMPUTE_BACKEND]),
        ),
    ],
    output: ("plan", "Typed fea.ExecutionPlan."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
