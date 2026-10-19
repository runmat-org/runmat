use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.TemperatureSignal.fromTimeKnots",
    summary: "Create a temperature time history from continuous and jump knots.",
    description: concat!(
        "`fea.TemperatureSignal.fromTimeKnots(id, knots, interpolation, extrapolation)` ",
        "returns an `fea.TemperatureSignal` whose value is a scalar in kelvin (K), absolute, ",
        "tabulated over time in seconds at the given knots.",
    ),
    keywords: &["fea", "signal", "time history", "temperature", "knots", "SI units"],
    related: &["fea.SignalKnot.continuous", "fea.SignalKnot.jump", "fea.SignalInterpolation.linear", "fea.SignalExtrapolation.holdEndpoint", "fea.TemperatureSignal.fromTimeSamples", "fea.ThermalExcitation.temperatureField", "fea.ThermalExcitation.ambientTemperature", "fea.ThermalInitialCondition.temperature", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                "The temperature ordinate of every knot is a scalar in kelvin (K), absolute.",
                super::HISTORY_BEHAVIOR,
                super::HISTORY_SAMPLING,
                super::HISTORY_RESULT,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Signals of this class feed `fea.ThermalExcitation.temperatureField`, ",
                    "`fea.ThermalExcitation.ambientTemperature`, ",
                    "`fea.ThermalInitialCondition.temperature`.",
                ),
                super::HISTORY_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "oven-ramp",
            "Heat an oven from 20 C to 100 C in 10 minutes",
            concat!(
                "knots = {fea.SignalKnot.continuous(0, 293.15), fea.SignalKnot.continuous(600, 373.15)};\n",
                "signal = fea.TemperatureSignal.fromTimeKnots('oven_ramp', knots, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.holdEndpoint());",
            ),
            concat!(
                "assert(strcmp(class(signal), 'fea.TemperatureSignal'));\n",
                "assert(signal.signal.temperature.signal_id == \"oven_ramp\");\n",
                "assert(signal.signal.temperature.shape == \"scalar\");\n",
                "assert(signal.signal.temperature.representation.tabulated.domain == \"time_seconds\");",
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
    FEA_TEMPERATURE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    name: "fea.TemperatureSignal.fromTimeKnots",
    result: classes::TEMPERATURE_SIGNAL,
    signature: concat!(
        "signal = fea.TemperatureSignal.fromTimeKnots(id, knots, interpolation, ",
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
                " in kelvin (K), absolute.",
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
    output: ("signal", "Typed fea.TemperatureSignal time history."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
