use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ComputeBackend.cpu",
    summary: "Request CPU execution for a procedure step.",
    description: "`fea.ComputeBackend.cpu()` returns the `fea.ComputeBackend` whose `value` is `cpu`.",
    keywords: &["fea", "execution", "backend", "cpu"],
    related: &["fea.ComputeBackend.gpu", "fea.ExecutionStep.new", "fea.ExecutionPlan.defaults", "fea.study"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The CPU backend runs the step's solver on the host. It is the backend `fea.study` ",
                    "uses when no execution plan is supplied.",
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
            "cpu",
            "Request CPU execution for a procedure step",
            "backend = fea.ComputeBackend.cpu();",
            concat!(
                "assert(strcmp(class(backend), 'fea.ComputeBackend'));\n",
                "assert(backend.value == \"cpu\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Is CPU always available?",
            "Yes. Every formulation that RunMat can execute has a host implementation.",
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
    FEA_COMPUTE_BACKEND_CPU_CATALOG_ENTRY,
    name: "fea.ComputeBackend.cpu",
    result: classes::COMPUTE_BACKEND,
    signature: "backend = fea.ComputeBackend.cpu()",
    inputs: [
    ],
    output: ("backend", "Typed fea.ComputeBackend value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
