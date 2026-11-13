use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ComputeBackend.gpu",
    summary: "Request GPU-accelerated execution for a procedure step.",
    description: "`fea.ComputeBackend.gpu()` returns the `fea.ComputeBackend` whose `value` is `gpu`.",
    keywords: &["fea", "execution", "backend", "gpu"],
    related: &["fea.ComputeBackend.cpu", "fea.ExecutionStep.new", "fea.ExecutionPlan.defaults", "fea.study"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The GPU backend asks the step's solver to use an accelerator. It is a request, not a",
                    " guarantee: when the solver for that formulation does not run on an accelerator, the",
                    " run executes on the host and records a solver-backend fallback event in its result ",
                    "evidence.",
                ),
                super::super::execution_plan::EXECUTION_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Pass the value as the `backend` argument of `fea.ExecutionStep.new` or ",
                    "`fea.ExecutionPlan.defaults`.",
                ),
                super::super::execution_plan::EXECUTION_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "gpu",
            "Request GPU-accelerated execution for a procedure step",
            "backend = fea.ComputeBackend.gpu();",
            concat!(
                "assert(strcmp(class(backend), 'fea.ComputeBackend'));\n",
                "assert(backend.value == \"gpu\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "What happens on a machine without a GPU?",
            concat!(
                "The study still runs. The result records the fallback so that you can see the step ",
                "did not execute on an accelerator.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/compute_backend/tests.rs::compute_backend_identities_conform",
        ),
        unit_test(
            "Default plans cover every sequence step",
            "crates/runmat-runtime/src/builtins/fea/execution.rs::typed_execution_plan_covers_each_sequence_step",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_COMPUTE_BACKEND_GPU_CATALOG_ENTRY,
    name: "fea.ComputeBackend.gpu",
    result: classes::COMPUTE_BACKEND,
    signature: "backend = fea.ComputeBackend.gpu()",
    inputs: [
    ],
    output: ("backend", "Typed fea.ComputeBackend value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
