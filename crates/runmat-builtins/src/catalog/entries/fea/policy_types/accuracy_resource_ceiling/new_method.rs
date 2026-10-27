use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.AccuracyResourceCeiling.new",
    summary: concat!(
        "Bound the automatic refinement, solver tightening, samples, wall time, and memory an",
        " accuracy plan may use.",
    ),
    description: concat!(
        "`fea.AccuracyResourceCeiling.new(maximumRefinementCycles, maximumSolverTightenings, ",
        "maximumSamples, maximumWallTimeMilliseconds, maximumResidentBytes)` returns the ",
        "resource ceiling of an accuracy plan. Its properties are ",
        "`maximum_refinement_cycles`, `maximum_solver_tightenings`, `maximum_samples`, ",
        "`maximum_wall_time_milliseconds`, and `maximum_resident_bytes`.",
    ),
    keywords: &["fea", "accuracy", "resources", "ceiling", "budget", "refinement"],
    related: &["fea.AccuracyPlan.new", "fea.AccuracyLimitAction.refine", "fea.AccuracyLimitAction.tightenSolver", "fea.ObservationCost.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "All five arguments are required and must be nonnegative integer values: ",
                    "integer-valued doubles or any built-in integer class, at most 2^53 - 1 ",
                    "(9007199254740991). Negative, fractional, nonfinite, or larger values fail with ",
                    "`RunMat:fea:InvalidInput`. Values within that bound are stored exactly.",
                ),
                concat!(
                    "Pass `[]` as `maximumResidentBytes` for no authored memory limit; the property is ",
                    "then empty and placement still limits the study to the capacity of the machine ",
                    "it runs on.",
                ),
                concat!(
                    "Wall time is in milliseconds (600000 is ten minutes) and memory is in bytes (2^32 is",
                    " 4 GiB).",
                ),
                concat!(
                    "Study validation then requires nonzero sample, wall-time, and resident-memory ",
                    "ceilings. Every observation on the step must fit: its declared `fea.ObservationCost`",
                    " peak resident bytes cannot exceed `maximumResidentBytes`, and its explicit or ",
                    "interval sample count cannot exceed `maximumSamples`.",
                ),
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Pass the ceiling as the last argument of `fea.AccuracyPlan.new`. The ceiling also ",
                    "licenses the plan's limit action: `fea.AccuracyLimitAction.refine()` requires ",
                    "`maximumRefinementCycles > 0` and `fea.AccuracyLimitAction.tightenSolver()` requires",
                    " `maximumSolverTightenings > 0`. A plan that names an action without the matching ",
                    "allowance fails validation.",
                ),
            ],
        ),
    ],
    examples: &[
        example(
            "ten-minute-budget",
            "Allow three refinements within ten minutes and 4 GiB",
            "resources = fea.AccuracyResourceCeiling.new(3, 2, 200, 600000, 2^32);",
            concat!(
                "assert(strcmp(class(resources), 'fea.AccuracyResourceCeiling'));\n",
                "assert(resources.maximum_refinement_cycles == 3);\n",
                "assert(resources.maximum_wall_time_milliseconds == 600000);\n",
                "assert(resources.maximum_resident_bytes == 4294967296);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Can I pass int64 or uint64 values?",
            concat!(
                "Yes. Every built-in integer class is accepted. Each ceiling must be at most 2^53 - 1 ",
                "(9007199254740991), the largest integer that study files and the browser represent ",
                "exactly; larger values fail with `RunMat:fea:InvalidInput`.",
            ),
        ),
        faq(
            "Is the wall time in seconds?",
            "No. It is in milliseconds. Write 600000 for ten minutes.",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/accuracy_resource_ceiling/tests.rs::accuracy_resource_ceiling_identities_conform",
        ),
        unit_test(
            "Resource ceilings stay exact within the wire bound",
            "crates/runmat-runtime/src/builtins/fea/accuracy.rs::resource_ceilings_stay_exact_within_the_wire_bound_and_reject_invalid_counts",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ACCURACY_RESOURCE_CEILING_NEW_CATALOG_ENTRY,
    name: "fea.AccuracyResourceCeiling.new",
    result: classes::ACCURACY_RESOURCE_CEILING,
    signature: concat!(
        "resources = fea.AccuracyResourceCeiling.new(maximumRefinementCycles, ",
        "maximumSolverTightenings, maximumSamples, maximumWallTimeMilliseconds, ",
        "maximumResidentBytes)",
    ),
    inputs: [
        (
            "maximumRefinementCycles",
            IntegerScalar,
            concat!(
                "Most automatic spatial refinement cycles the plan may request; 0 disables ",
                "refinement.",
            ),
            PolicyArg::Count,
        ),
        (
            "maximumSolverTightenings",
            IntegerScalar,
            "Most automatic solver-tolerance tightenings; 0 disables tightening.",
            PolicyArg::Count,
        ),
        (
            "maximumSamples",
            IntegerScalar,
            concat!(
                "Most observation or study samples any one observation may request; must be nonzero ",
                "at validation.",
            ),
            PolicyArg::Count,
        ),
        (
            "maximumWallTimeMilliseconds",
            IntegerScalar,
            "Wall-time ceiling in milliseconds; must be nonzero at validation.",
            PolicyArg::Count,
        ),
        (
            "maximumResidentBytes",
            IntegerScalar,
            "Resident-memory ceiling in bytes; must be nonzero at validation.",
            PolicyArg::Count,
        ),
    ],
    output: ("resources", "Typed fea.AccuracyResourceCeiling."),
    integer_capabilities: &integer::RESOURCE_CEILING_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
