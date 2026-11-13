use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyPlan.new",
    summary: "Create the accuracy acceptance plan of one procedure step.",
    description: concat!(
        "`fea.AccuracyPlan.new(id, criteria, unavailableEvidence, exceededLimit, equivalence,",
        " resources)` returns an `fea.AccuracyPlan` with properties `accuracy_plan_id`, ",
        "`criteria`, `unavailable_evidence`, `exceeded_limit`, `equivalence`, and ",
        "`resources`.",
    ),
    keywords: &["fea", "accuracy", "plan", "acceptance", "evidence", "resources"],
    related: &["fea.AccuracyCriterion.new", "fea.AccuracyResourceCeiling.new", "fea.NumericalEquivalencePolicy.relativeTolerance", "fea.AccuracyPlan.baseline", "fea.ProcedureStep.configure"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The constructor checks argument types: a nonempty identifier, a criterion or cell ",
                    "array of criteria, and one value of each typed policy class. It does not check the ",
                    "plan's internal consistency, so an empty criteria cell is accepted here and rejected",
                    " at study validation.",
                ),
                concat!(
                    "Study validation requires at least one criterion, at least one mandatory criterion, ",
                    "unique criterion identifiers, positive tolerances, a positive equivalence tolerance,",
                    " and nonzero sample, wall-time, and memory ceilings. The limit action must be ",
                    "licensed: `refine` needs refinement cycles and `tightenSolver` needs solver ",
                    "tightenings.",
                ),
                concat!(
                    "When a mandatory criterion is not established, the step's semantic state is not ",
                    "published. With `fea.UnavailableEvidencePolicy.markIndeterminate()`, ",
                    "missing evidence is reported as indeterminate instead of failing.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                super::PLAN_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "static-acceptance",
            "Accept a static tip displacement within 1 percent",
            concat!(
                "criterion = fea.AccuracyCriterion.new('tip_mesh', ...\n",
                "    fea.AccuracyTarget.observation('tip_displacement'), ...\n",
                "    fea.ErrorContribution.spatialDiscretization(), ...\n",
                "    fea.AccuracyTolerance.relative(0.01), ...\n",
                "    fea.RequiredAccuracyEvidence.convergenceStudy(), true);\n",
                "plan = fea.AccuracyPlan.new('static_acceptance', {criterion}, ...\n",
                "    fea.UnavailableEvidencePolicy.fail(), ...\n",
                "    fea.AccuracyLimitAction.refine(), ...\n",
                "    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), ...\n",
                "    fea.AccuracyResourceCeiling.new(3, 2, 100, 600000, 2000000000));",
            ),
            concat!(
                "assert(strcmp(class(plan), 'fea.AccuracyPlan'));\n",
                "assert(plan.accuracy_plan_id == \"static_acceptance\");\n",
                "assert(numel(plan.criteria) == 1);\n",
                "assert(plan.exceeded_limit == \"refine\");\n",
                "assert(plan.unavailable_evidence == \"fail\");",
            ),
        ),
    ],
    faqs: &[
        super::PLAN_FAQ,
        faq(
            "Can one plan hold criteria for several error sources?",
            concat!(
                "Yes. Use one criterion per error source, for example spatial discretization of a ",
                "displacement and an algebraic residual bound, and pass them as a cell array.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_plan/tests.rs::accuracy_plan_identities_conform",
        ),
        unit_test(
            "Plans accept criteria cells and round-trip portably",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::plans_accept_criteria_cells_and_round_trip_portably",
        ),
        unit_test(
            "Accuracy plan validation rejects nonpositive tolerances",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_nonpositive_tolerance",
        ),
        unit_test(
            "Observation targets must exist on the step",
            "crates/runmat-analysis/core/src/validate_accuracy.rs::rejects_unresolved_observation_target",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_PLAN_NEW_CATALOG_ENTRY,
    name: "fea.AccuracyPlan.new",
    result: classes::ACCURACY_PLAN,
    signature: concat!(
        "plan = fea.AccuracyPlan.new(id, criteria, unavailableEvidence, exceededLimit, ",
        "equivalence, resources)",
    ),
    inputs: [
        (
            "id",
            StringScalar,
            "Nonempty accuracy-plan identifier.",
            PolicyArg::Text,
        ),
        (
            "criteria",
            Any,
            "One fea.AccuracyCriterion or a cell array of them.",
            PolicyArg::TypedOrCell(&[classes::ACCURACY_CRITERION]),
        ),
        (
            "unavailableEvidence",
            Any,
            concat!(
                "Typed fea.UnavailableEvidencePolicy: what happens when required evidence cannot be ",
                "produced.",
            ),
            PolicyArg::Typed(&[classes::UNAVAILABLE_EVIDENCE_POLICY]),
        ),
        (
            "exceededLimit",
            Any,
            concat!(
                "Typed fea.AccuracyLimitAction: what happens when a criterion is not met within the ",
                "resource ceiling.",
            ),
            PolicyArg::Typed(&[classes::ACCURACY_LIMIT_ACTION]),
        ),
        (
            "equivalence",
            Any,
            "Typed fea.NumericalEquivalencePolicy for agreement between executions.",
            PolicyArg::Typed(&[classes::NUMERICAL_EQUIVALENCE_POLICY]),
        ),
        (
            "resources",
            Any,
            "Typed fea.AccuracyResourceCeiling.",
            PolicyArg::Typed(&[classes::ACCURACY_RESOURCE_CEILING]),
        ),
    ],
    output: ("plan", "Typed fea.AccuracyPlan."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
