use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ExecutionStep.new",
    summary: "Bind one procedure step to a compute backend, its accelerator admission, and solver options.",
    description: concat!(
        "`fea.ExecutionStep.new(step, backend, solverOptions)` or ",
        "`fea.ExecutionStep.new(step, backend, solverOptions, strict)` returns an ",
        "`fea.ExecutionStep` with properties `step_id`, `backend`, `accelerator_admission`, and ",
        "`solver_options`. Only the step's identifier is retained; the step itself stays in ",
        "the analysis sequence.",
    ),
    keywords: &["fea", "execution", "step", "backend", "solver"],
    related: &["fea.SolverOptions.defaults", "fea.ComputeBackend.cpu", "fea.ExecutionPlan.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Every argument must be of its typed class. The solver options must support the ",
                    "step's family, procedure, and solution space; options selected for another procedure",
                    " fail with `RunMat:fea:InvalidInput` (\"solver options do not support the supplied ",
                    "procedure step\").",
                ),
                super::super::execution_plan::EXECUTION_ROLE,
            ],
        ),
        section(
            "GPU fallback and strict placement",
            &[
                concat!(
                    "`strict` is optional and defaults to `false`. ",
                    "With `strict = false`, a step that requests `fea.ComputeBackend.gpu()` but cannot ",
                    "run on the accelerator (its solver has no device path, or no accelerator is ",
                    "assigned) runs on the CPU reference path: the result reports `cpu_reference` as ",
                    "its execution backend and carries one `FEA_ACCELERATOR_ADMISSION_FALLBACK` ",
                    "warning. `accelerator_admission` is then `fallback_to_cpu`.",
                ),
                concat!(
                    "With `strict = true` (`accelerator_admission = required`, the same policy as ",
                    "`accelerators = required` in a `.fea` execution policy), the step fails closed ",
                    "with `RM.FEA.ACCELERATOR_REQUIRED` instead of falling back. When the fallback is ",
                    "known before the solve, the step fails before any assembly.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Create one execution step for each procedure step of a heterogeneous sequence, then ",
                    "pass them in a cell array to `fea.ExecutionPlan.new(sequence, steps)`. When every ",
                    "step can use the same backend and default options, ",
                    "`fea.ExecutionPlan.defaults` is shorter.",
                ),
                super::super::execution_plan::EXECUTION_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "gpu-static",
            "Request GPU execution of a static step",
            concat!(
                "step = fea.StructuralStep.static('preload');\n",
                "execution = fea.ExecutionStep.new(step, fea.ComputeBackend.gpu(), ...\n",
                "    fea.SolverOptions.defaults(step));",
            ),
            concat!(
                "assert(strcmp(class(execution), 'fea.ExecutionStep'));\n",
                "assert(execution.step_id == \"preload\");\n",
                "assert(execution.backend == \"gpu\");\n",
                "assert(execution.accelerator_admission == \"fallback_to_cpu\");",
            ),
        ),
        example(
            "gpu-static-strict",
            "Require the accelerator for a static step",
            concat!(
                "step = fea.StructuralStep.static('preload');\n",
                "execution = fea.ExecutionStep.new(step, fea.ComputeBackend.gpu(), ...\n",
                "    fea.SolverOptions.defaults(step), true);",
            ),
            "assert(execution.accelerator_admission == \"required\");",
        ),
    ],
    faqs: &[
        faq(
            "Why does the execution step store only the step identifier?",
            concat!(
                "The analysis sequence remains the single authority for the step's physics. The ",
                "execution plan binds to it by identifier and is checked against the sequence when ",
                "the plan is built.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/execution_step/tests.rs::execution_step_identities_conform",
        ),
        unit_test(
            "Solver options must match their procedure",
            "crates/runmat-runtime/src/builtins/fea/execution.rs::execution_step_rejects_solver_options_for_another_procedure",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_EXECUTION_STEP_NEW_CATALOG_ENTRY,
    name: "fea.ExecutionStep.new",
    result: classes::EXECUTION_STEP,
    signature: "execution = fea.ExecutionStep.new(step, backend, solverOptions, strict)",
    inputs: [
        (
            "step",
            Any,
            "One typed procedure step, or a one-element cell holding it.",
            PolicyArg::TypedOrCell(&classes::PROCEDURE_STEP_CLASSES),
        ),
        (
            "backend",
            Any,
            "Typed fea.ComputeBackend.",
            PolicyArg::Typed(&[classes::COMPUTE_BACKEND]),
        ),
        (
            "solverOptions",
            Any,
            "Typed fea.SolverOptions compatible with the step's procedure.",
            PolicyArg::Typed(&[classes::SOLVER_OPTIONS]),
        ),
        (
            "strict",
            LogicalArray,
            "Optional logical scalar (default false); true fails closed when a GPU step cannot run on the accelerator.",
            PolicyArg::Optional(&PolicyArg::Logical),
            default: "false",
        ),
    ],
    output: ("execution", "Typed fea.ExecutionStep."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
