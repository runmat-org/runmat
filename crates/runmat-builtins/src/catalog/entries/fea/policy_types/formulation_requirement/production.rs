use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.FormulationRequirement.production",
    summary: "Admit only production-qualified formulations for a procedure step.",
    description: concat!(
        "`fea.FormulationRequirement.production()` returns the `fea.FormulationRequirement` ",
        "whose `value` is `production`.",
    ),
    keywords: &["fea", "execution", "requirement", "production"],
    related: &["fea.FormulationRequirement.experimental", "fea.ExecutionStep.new", "fea.ExecutionPlan.defaults", "fea.study"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "With the production requirement, a run fails at the pre-solve admission boundary if ",
                    "the step's physics family, procedure, and solution space have no ",
                    "production-qualified formulation. Nothing is solved and no result is published.",
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
            "production",
            "Admit only production-qualified formulations for a procedure step",
            "requirement = fea.FormulationRequirement.production();",
            concat!(
                "assert(strcmp(class(requirement), 'fea.FormulationRequirement'));\n",
                "assert(requirement.value == \"production\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Why did my study fail admission with production?",
            concat!(
                "The formulation for that procedure is still experimental. Use ",
                "`fea.FormulationRequirement.experimental()` to run it explicitly as experimental ",
                "work.",
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
    FEA_FORMULATION_REQUIREMENT_PRODUCTION_CATALOG_ENTRY,
    name: "fea.FormulationRequirement.production",
    result: classes::FORMULATION_REQUIREMENT,
    signature: "requirement = fea.FormulationRequirement.production()",
    inputs: [
    ],
    output: ("requirement", "Typed fea.FormulationRequirement value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
