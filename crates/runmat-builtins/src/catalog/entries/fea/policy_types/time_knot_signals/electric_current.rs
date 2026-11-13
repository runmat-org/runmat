use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ElectricCurrentSignal.fromTimeKnots",
    summary: "Create a electric current time history from continuous and jump knots.",
    description: concat!(
        "`fea.ElectricCurrentSignal.fromTimeKnots(id, knots, interpolation, extrapolation)` ",
        "returns an `fea.ElectricCurrentSignal` whose value is a scalar in amperes (A), ",
        "tabulated over time in seconds at the given knots.",
    ),
    keywords: &["fea", "signal", "time history", "electric current", "knots", "SI units"],
    related: &["fea.SignalKnot.continuous", "fea.SignalKnot.jump", "fea.SignalInterpolation.linear", "fea.SignalExtrapolation.holdEndpoint", "fea.ElectricCurrentSignal.fromTimeSamples", "fea.ElectricalExcitation.current", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                "The electric current ordinate of every knot is a scalar in amperes (A).",
                super::HISTORY_BEHAVIOR,
                super::HISTORY_SAMPLING,
                super::HISTORY_RESULT,
            ],
        ),
        section(
            "Use in a study",
            &[
                "Signals of this class feed `fea.ElectricalExcitation.current`.",
                super::HISTORY_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "drive-current",
            "Drive a 10 A current step",
            concat!(
                "knots = {fea.SignalKnot.continuous(0, 0), fea.SignalKnot.jump(0.002, 0, 10, fea.SignalPointContinuity.right()), fea.SignalKnot.continuous(0.02, 10)};\n",
                "signal = fea.ElectricCurrentSignal.fromTimeKnots('drive_current', knots, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.holdEndpoint());",
            ),
            concat!(
                "assert(strcmp(class(signal), 'fea.ElectricCurrentSignal'));\n",
                "assert(signal.signal.electric_current.signal_id == \"drive_current\");\n",
                "assert(signal.signal.electric_current.shape == \"scalar\");\n",
                "assert(signal.signal.electric_current.representation.tabulated.domain == \"time_seconds\");",
            ),
        ),
    ],
    faqs: &[
        super::HISTORY_FAQ,
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/time_knot_signals/tests.rs::time_knot_signal_identities_conform",
        ),
        unit_test(
            "Extrapolation policy is applied without numeric sentinels",
            "crates/runmat-analysis/fea/src/excitation/tests.rs::extrapolation_policy_is_applied_without_numeric_sentinels",
        ),
        unit_test(
            "Jump knots have distinct point and one-sided values",
            "crates/runmat-analysis/fea/src/excitation/tests.rs::jump_knots_have_distinct_point_and_one_sided_values",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_ELECTRIC_CURRENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    name: "fea.ElectricCurrentSignal.fromTimeKnots",
    result: classes::ELECTRIC_CURRENT_SIGNAL,
    signature: concat!(
        "signal = fea.ElectricCurrentSignal.fromTimeKnots(id, knots, interpolation, ",
        "extrapolation)",
    ),
    inputs: [
        (
            "id",
            StringScalar,
            "Stable signal identifier.",
            PolicyArg::Text,
        ),
        (
            "knots",
            Any,
            concat!(
                "Row or column cell array of at least two fea.SignalKnot values with scalar ordinates",
                " in amperes (A).",
            ),
            PolicyArg::TypedCell { classes: &[classes::SIGNAL_KNOT], minimum: 2 },
        ),
        (
            "interpolation",
            Any,
            "Typed fea.SignalInterpolation between knots.",
            PolicyArg::Typed(&[classes::SIGNAL_INTERPOLATION]),
        ),
        (
            "extrapolation",
            Any,
            "Typed fea.SignalExtrapolation outside the first and last knot times.",
            PolicyArg::Typed(&[classes::SIGNAL_EXTRAPOLATION]),
        ),
    ],
    output: ("signal", "Typed fea.ElectricCurrentSignal time history."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
