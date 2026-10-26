use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyCriterion.new",
    summary: concat!(
        "Create one accuracy acceptance criterion: a target, error source, tolerance, and ",
        "required evidence.",
    ),
    description: concat!(
        "`fea.AccuracyCriterion.new(id, target, contribution, tolerance, evidence, ",
        "mandatory)` returns an `fea.AccuracyCriterion` with properties `criterion_id`, ",
        "`target`, `contribution`, `tolerance`, `evidence`, and `mandatory`.",
    ),
    keywords: &["fea", "accuracy", "criterion", "acceptance", "evidence", "tolerance"],
    related: &["fea.AccuracyTarget.observation", "fea.AccuracyTolerance.relative", "fea.ErrorContribution.spatialDiscretization", "fea.RequiredAccuracyEvidence.convergenceStudy", "fea.AccuracyPlan.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Each argument is checked when the criterion is created: `id` must be nonempty text, ",
                    "the four policy arguments must be values of their typed classes, and `mandatory` ",
                    "must be a logical scalar. Numeric `1` or `0` is not accepted for `mandatory`; use ",
                    "`true` or `false`. Any failure raises `RunMat:fea:InvalidInput` and returns nothing.",
                ),
                concat!(
                    "Error contributions stay separate rather than being summed into one total, so one ",
                    "criterion bounds exactly one source of error, such as spatial discretization, ",
                    "temporal integration, or modal truncation.",
                ),
                concat!(
                    "Compatibility is checked when the study is validated. The contribution must suit the",
                    " target (global equilibrium accepts algebraic, nonlinear, or coupling error; ",
                    "interface closure accepts coupling, interface-projection, or field-transfer error; ",
                    "execution equivalence accepts only execution-equivalence error). The evidence must ",
                    "be able to establish the contribution; for example, spatial discretization needs an ",
                    "a-posteriori estimator, convergence study, or reference comparison, while temporal ",
                    "and modal-truncation error need a convergence study or reference comparison. ",
                    "Criterion identifiers must be unique within the plan.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Collect criteria in the second argument of `fea.AccuracyPlan.new`, as a single ",
                    "criterion or a cell array. A plan needs at least one mandatory criterion. Optional ",
                    "criteria are assessed and reported but do not block acceptance.",
                ),
            ],
        ),
    ],
    examples: &[
        example(
            "tip-displacement-mesh",
            "Require a mesh convergence study for the tip displacement",
            concat!(
                "criterion = fea.AccuracyCriterion.new('tip_mesh', ...\n",
                "    fea.AccuracyTarget.observation('tip_displacement'), ...\n",
                "    fea.ErrorContribution.spatialDiscretization(), ...\n",
                "    fea.AccuracyTolerance.relative(0.01), ...\n",
                "    fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            ),
            concat!(
                "assert(strcmp(class(criterion), 'fea.AccuracyCriterion'));\n",
                "assert(criterion.criterion_id == \"tip_mesh\");\n",
                "assert(criterion.mandatory);\n",
                "assert(criterion.contribution == \"spatial_discretization\");\n",
                "assert(criterion.evidence == \"convergence_study\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Why does mandatory reject 1?",
            "Typed FEA policies do not reinterpret numbers as flags. Pass `true` or `false`.",
        ),
        faq(
            "When are incompatible combinations reported?",
            concat!(
                "When the study is validated, before any solve starts. The constructor only checks ",
                "the type of each argument.",
            ),
        ),
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_criterion/tests.rs::accuracy_criterion_identities_conform",
        ),
        unit_test(
            "Criteria require typed policies and a logical flag",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::criterion_requires_typed_policies_and_a_logical_mandatory_flag",
        ),
        unit_test(
            "Recovery evidence compatibility",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::a_posteriori_recovery_evidence_is_valid_without_relabeling_transfer",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_CRITERION_NEW_CATALOG_ENTRY,
    name: "fea.AccuracyCriterion.new",
    result: classes::ACCURACY_CRITERION,
    signature: concat!(
        "criterion = fea.AccuracyCriterion.new(id, target, contribution, tolerance, evidence,",
        " mandatory)",
    ),
    inputs: [
        (
            "id",
            StringScalar,
            "Nonempty criterion identifier, unique within its accuracy plan.",
            PolicyArg::Text,
        ),
        (
            "target",
            Any,
            "Typed fea.AccuracyTarget: what the criterion measures.",
            PolicyArg::Typed(&[classes::ACCURACY_TARGET]),
        ),
        (
            "contribution",
            Any,
            "Typed fea.ErrorContribution: which error source the criterion bounds.",
            PolicyArg::Typed(&[classes::ERROR_CONTRIBUTION]),
        ),
        (
            "tolerance",
            Any,
            "Typed fea.AccuracyTolerance: the acceptance bound.",
            PolicyArg::Typed(&[classes::ACCURACY_TOLERANCE]),
        ),
        (
            "evidence",
            Any,
            "Typed fea.RequiredAccuracyEvidence: how the bound must be demonstrated.",
            PolicyArg::Typed(&[classes::REQUIRED_ACCURACY_EVIDENCE]),
        ),
        (
            "mandatory",
            LogicalArray,
            "Logical scalar; true makes failure block acceptance of the step.",
            PolicyArg::Logical,
        ),
    ],
    output: ("criterion", "Typed fea.AccuracyCriterion."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
