use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.TemperatureDifferenceSignal.fromTimeKnots",
    summary: "Create a temperature difference time history from continuous and jump knots.",
    description: concat!(
        "`fea.TemperatureDifferenceSignal.fromTimeKnots(id, knots, interpolation, ",
        "extrapolation)` returns an `fea.TemperatureDifferenceSignal` whose value is a scalar",
        " in kelvin (K), as a difference, tabulated over time in seconds at the given knots.",
    ),
    keywords: &["fea", "signal", "time history", "temperature difference", "knots", "SI units"],
    related: &["fea.SignalKnot.continuous", "fea.SignalKnot.jump", "fea.SignalInterpolation.linear", "fea.SignalExtrapolation.holdEndpoint", "fea.TemperatureDifferenceSignal.fromTimeSamples", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The temperature difference ordinate of every knot is a scalar in kelvin (K), as a ",
                    "difference.",
                ),
                super::HISTORY_BEHAVIOR,
                super::HISTORY_SAMPLING,
                super::HISTORY_RESULT,
            ],
        ),
        section(
            "Use in a study",
            &[
                "Use the signal wherever a model object consumes a temperature difference signal.",
                super::HISTORY_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "rise",
            "Ramp a temperature rise of 80 K",
            concat!(
                "knots = {fea.SignalKnot.continuous(0, 0), fea.SignalKnot.continuous(600, 80)};\n",
                "signal = fea.TemperatureDifferenceSignal.fromTimeKnots('rise', knots, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.holdEndpoint());",
            ),
            concat!(
                "assert(strcmp(class(signal), 'fea.TemperatureDifferenceSignal'));\n",
                "assert(signal.signal.temperature_difference.signal_id == \"rise\");\n",
                "assert(signal.signal.temperature_difference.shape == \"scalar\");\n",
                "assert(signal.signal.temperature_difference.representation.tabulated.domain == \"time_seconds\");",
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
    FEA_TEMPERATURE_DIFFERENCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    name: "fea.TemperatureDifferenceSignal.fromTimeKnots",
    result: classes::TEMPERATURE_DIFFERENCE_SIGNAL,
    signature: concat!(
        "signal = fea.TemperatureDifferenceSignal.fromTimeKnots(id, knots, interpolation, ",
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
                " in kelvin (K), as a difference.",
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
    output: ("signal", "Typed fea.TemperatureDifferenceSignal time history."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
