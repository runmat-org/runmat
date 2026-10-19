use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.FormulationRequirement.experimental",
    summary: "Admit experimental formulations for a procedure step.",
    description: concat!(
        "`fea.FormulationRequirement.experimental()` returns the `fea.FormulationRequirement`",
        " whose `value` is `experimental`.",
    ),
    keywords: &["fea", "execution", "requirement", "experimental"],
    related: &["fea.FormulationRequirement.production", "fea.AccuracyPlan.experimentalBaseline", "fea.ExecutionStep.new", "fea.ExecutionPlan.defaults", "fea.study"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "With the experimental requirement, formulations that have not yet earned production ",
                    "qualification may run. The result stays marked by the qualification of the ",
                    "formulation that produced it; choosing experimental never relabels a result as ",
                    "production.",
                ),
                super::super::execution_plan::EXECUTION_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Pass the value as the `requirement` argument of `fea.ExecutionStep.new` or ",
                    "`fea.ExecutionPlan.defaults`.",
                ),
                super::super::execution_plan::EXECUTION_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "experimental",
            "Admit experimental formulations for a procedure step",
            "requirement = fea.FormulationRequirement.experimental();",
            concat!(
                "assert(strcmp(class(requirement), 'fea.FormulationRequirement'));\n",
                "assert(requirement.value == \"experimental\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Does experimental change the numerical method?",
            concat!(
                "No. It only widens which formulations the caller accepts. Accuracy plans and result ",
                "qualification are evaluated the same way.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/formulation_requirement/tests.rs::formulation_requirement_identities_conform",
        ),
        unit_test(
            "Default plans cover every sequence step",
            "crates/runmat-runtime/src/builtins/fea/execution.rs::typed_execution_plan_covers_each_sequence_step",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_FORMULATION_REQUIREMENT_EXPERIMENTAL_CATALOG_ENTRY,
    name: "fea.FormulationRequirement.experimental",
    result: classes::FORMULATION_REQUIREMENT,
    signature: "requirement = fea.FormulationRequirement.experimental()",
    inputs: [
    ],
    output: ("requirement", "Typed fea.FormulationRequirement value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
