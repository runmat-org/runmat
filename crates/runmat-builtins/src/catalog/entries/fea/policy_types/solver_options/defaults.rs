use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SolverOptions.defaults",
    summary: "Select the default solver options for a typed procedure step.",
    description: concat!(
        "`fea.SolverOptions.defaults(step)` returns the `fea.SolverOptions` RunMat uses for ",
        "`step` when no options are authored. The `value` names the selected solver (for ",
        "example `structural_linear` or `thermal_diffusion`) and its default settings.",
    ),
    keywords: &["fea", "execution", "solver", "options", "defaults"],
    related: &["fea.ExecutionStep.new", "fea.ExecutionPlan.defaults", "fea.StructuralStep.static", "fea.ThermalStep.steady"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The solver is chosen from the step's physics family, procedure, and solution space: ",
                    "structural static steps get linear-structural options, modal steps get ",
                    "modal-extraction options sized to the step's requested mode count, thermal steady ",
                    "and transient steps get thermal-diffusion options, and so on for harmonic, ",
                    "response-spectrum, random-response, buckling, transient, fluid, conjugate heat ",
                    "transfer, fluid-structure, and nonlinear steps.",
                ),
                concat!(
                    "A procedure without a default fails with `RunMat:fea:InvalidInput`. Static ",
                    "electromagnetic steps, for example, require explicitly authored equation-bound ",
                    "solver options, and a modal step must have an eigenvalue domain. A cell holding more",
                    " or fewer than one step is also rejected.",
                ),
                super::super::execution_plan::EXECUTION_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Pass the options to `fea.ExecutionStep.new` with the same step. Options carry the ",
                    "solver they were selected for, so options created for one procedure are rejected ",
                    "when bound to a step of another procedure.",
                ),
                super::super::execution_plan::EXECUTION_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "thermal-steady",
            "Select default options for a steady thermal step",
            concat!(
                "step = fea.ThermalStep.steady('heat_soak');\n",
                "options = fea.SolverOptions.defaults(step);",
            ),
            concat!(
                "assert(strcmp(class(options), 'fea.SolverOptions'));\n",
                "assert(options.value.solver == \"thermal_diffusion\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Can I tune individual solver settings?",
            concat!(
                "This constructor returns the defaults for the step. Bind them with ",
                "`fea.ExecutionStep.new`; the options value is immutable.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/solver_options/tests.rs::solver_options_identities_conform",
        ),
        unit_test(
            "Solver options must match their procedure",
            "crates/runmat-runtime/src/builtins/fea/execution.rs::execution_step_rejects_solver_options_for_another_procedure",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SOLVER_OPTIONS_DEFAULTS_CATALOG_ENTRY,
    name: "fea.SolverOptions.defaults",
    result: classes::SOLVER_OPTIONS,
    signature: "options = fea.SolverOptions.defaults(step)",
    inputs: [
        (
            "step",
            Any,
            concat!(
                "One typed procedure step (for example fea.StructuralStep.static), or a one-element ",
                "cell holding it.",
            ),
            PolicyArg::TypedOrCell(&classes::PROCEDURE_STEP_CLASSES),
        ),
    ],
    output: ("options", "Typed fea.SolverOptions selected for the step's procedure."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
