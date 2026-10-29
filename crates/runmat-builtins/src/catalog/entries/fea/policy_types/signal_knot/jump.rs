use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalKnot.jump",
    summary: "Create a knot where a time history jumps, with explicit one-sided limits.",
    description: concat!(
        "`fea.SignalKnot.jump(abscissa, leftLimit, rightLimit, continuity)` returns an ",
        "`fea.SignalKnot` that represents a discontinuity at time `abscissa`. The ",
        "`knot.value.Jump` struct holds `left_limit`, `right_limit`, and `continuity`.",
    ),
    keywords: &["fea", "signal", "knot", "jump", "discontinuity", "step load"],
    related: &["fea.SignalKnot.continuous", "fea.SignalPointContinuity.left", "fea.SignalPointContinuity.right", "fea.ForceSignal.fromTimeKnots"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "A jump has three values. Approaching from earlier times the signal tends to ",
                    "`leftLimit`; approaching from later times it tends to `rightLimit`. The value at the",
                    " abscissa itself is `leftLimit` when `continuity` is ",
                    "`fea.SignalPointContinuity.left()` and `rightLimit` when it is ",
                    "`fea.SignalPointContinuity.right()`.",
                ),
                concat!(
                    "Interpolation on the interval before the jump ends at `leftLimit`, and interpolation",
                    " on the interval after it starts at `rightLimit`, so linear interpolation never ",
                    "smears the step. A solver that samples the one-sided limits at the jump time sees ",
                    "both values; RunMat's transient solvers use this to place an exact event at the ",
                    "discontinuity.",
                ),
                concat!(
                    "`leftLimit` and `rightLimit` must have the same shape (both scalar or both ",
                    "three-component); a mismatch fails with `RunMat:fea:InvalidInput`. `continuity` must",
                    " be a typed `fea.SignalPointContinuity`.",
                ),
                super::KNOT_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "A jump must be interior to its signal: the first and last knots of a `fromTimeKnots`",
                    " history must be continuous. Put at least one continuous knot before and after each ",
                    "jump.",
                ),
                super::KNOT_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "step-load",
            "Apply a 1500 Pa step at 10 ms",
            concat!(
                "before = fea.SignalKnot.continuous(0, 0);\n",
                "step = fea.SignalKnot.jump(0.01, 0, 1500, fea.SignalPointContinuity.right());\n",
                "after = fea.SignalKnot.continuous(0.05, 1500);",
            ),
            concat!(
                "assert(strcmp(class(step), 'fea.SignalKnot'));\n",
                "assert(step.knot.value.Jump.left_limit == 0);\n",
                "assert(step.knot.value.Jump.right_limit == 1500);\n",
                "assert(step.knot.value.Jump.continuity == \"right\");",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Which continuity should a step load use?",
            concat!(
                "Use `right()` when the new value applies at the step time itself, the usual ",
                "convention for a load switched on at that instant. Use `left()` when the old value ",
                "still holds at the step time.",
            ),
        ),
        faq(
            "Can a signal start with a jump?",
            concat!(
                "No. Jumps must be interior; a history whose first or last knot is a jump is rejected",
                " by `fromTimeKnots`.",
            ),
        ),
    ],
    links: &[
        page("Loads, excitations, and natural boundaries", "/docs/simulation/loads"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/signals/knots.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/signal_knot/tests.rs::signal_knot_identities_conform",
        ),
        unit_test(
            "Authored zero is canonical and payloads are portable",
            "crates/runmat-runtime/src/builtins/fea/signals/knots.rs::portable_knots_normalize_authored_zero_and_reject_noncanonical_payloads",
        ),
        unit_test(
            "Jump knots have distinct point and one-sided values",
            "crates/runmat-analysis/fea/src/excitation/tests.rs::jump_knots_have_distinct_point_and_one_sided_values",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SIGNAL_KNOT_JUMP_CATALOG_ENTRY,
    name: "fea.SignalKnot.jump",
    result: classes::SIGNAL_KNOT,
    signature: "knot = fea.SignalKnot.jump(abscissa, leftLimit, rightLimit, continuity)",
    inputs: [
        (
            "abscissa",
            NumericScalar,
            "Finite time of the discontinuity in seconds.",
            PolicyArg::RealScalar,
        ),
        (
            "leftLimit",
            NumericArray,
            "Value approached from earlier times, in the owning signal's SI unit.",
            PolicyArg::ScalarOrVector3,
        ),
        (
            "rightLimit",
            NumericArray,
            "Value approached from later times; same shape as leftLimit.",
            PolicyArg::ScalarOrVector3,
        ),
        (
            "continuity",
            Any,
            "Typed fea.SignalPointContinuity selecting which limit is the value at the abscissa.",
            PolicyArg::Typed(&[classes::SIGNAL_POINT_CONTINUITY]),
        ),
    ],
    output: ("knot", "Typed fea.SignalKnot."),
    integer_capabilities: &integer::JUMP_KNOT_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
