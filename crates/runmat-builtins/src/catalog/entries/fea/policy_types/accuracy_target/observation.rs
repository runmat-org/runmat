use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyTarget.observation",
    summary: "Target one named observation of a procedure step for accuracy acceptance.",
    description: concat!(
        "`fea.AccuracyTarget.observation(id)` returns an `fea.AccuracyTarget` that points an ",
        "accuracy criterion at the observation whose identifier is `id`. The stored value is ",
        "the struct `observation: id`.",
    ),
    keywords: &["fea", "accuracy", "target", "observation", "acceptance"],
    related: &["fea.Observation.field", "fea.Observation.history", "fea.AccuracyTarget.quantity", "fea.AccuracyCriterion.new", "fea.AccuracyPlan.new", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`id` must be nonempty text; an empty or whitespace-only identifier fails immediately",
                    " with `RunMat:fea:InvalidInput`. The identifier is stored exactly as written and is ",
                    "not resolved when the target is created.",
                ),
                concat!(
                    "When the study is validated, the identifier must name a field, history, probe, or ",
                    "reduction observation on the same procedure step. Diagnostics and monitors cannot be",
                    " accuracy targets because they do not carry a numeric physical result. If the ",
                    "criterion uses an absolute tolerance, the tolerance quantity must equal the ",
                    "observation's quantity.",
                ),
                super::TARGET_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Use an observation target when acceptance depends on one reported result, such as a ",
                    "tip displacement or a peak stress, rather than on the whole solution. Declare the ",
                    "observation with `fea.Observation.field`, `history`, `probe`, or `reduction`, and ",
                    "pass it to the same `fea.ProcedureStep.configure` call as the accuracy plan.",
                ),
                super::TARGET_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "tip-displacement",
            "Accept a tip displacement within 1 percent",
            concat!(
                "target = fea.AccuracyTarget.observation('tip_displacement');\n",
                "criterion = fea.AccuracyCriterion.new('tip_mesh', target, ...\n",
                "    fea.ErrorContribution.spatialDiscretization(), ...\n",
                "    fea.AccuracyTolerance.relative(0.01), ...\n",
                "    fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            ),
            concat!(
                "assert(strcmp(class(target), 'fea.AccuracyTarget'));\n",
                "assert(target.value.observation == \"tip_displacement\");\n",
                "assert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
            ),
        ),
    ],
    faqs: &[
        faq(
            "When is the observation identifier checked?",
            concat!(
                "At study validation. The constructor only rejects an empty identifier, so a ",
                "criterion can be authored before or after the observation it names, as long as both ",
                "end up on the same procedure step.",
            ),
        ),
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
            "Observation targets must exist on the step",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_unresolved_observation_target",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_TARGET_OBSERVATION_CATALOG_ENTRY,
    name: "fea.AccuracyTarget.observation",
    result: classes::ACCURACY_TARGET,
    signature: "target = fea.AccuracyTarget.observation(id)",
    inputs: [
        (
            "id",
            StringScalar,
            "Identifier of a numeric fea.Observation declared on the same procedure step.",
            PolicyArg::Text,
        ),
    ],
    output: ("target", "Typed fea.AccuracyTarget naming one observation."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
