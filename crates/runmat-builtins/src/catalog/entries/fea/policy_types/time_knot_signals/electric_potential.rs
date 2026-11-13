use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ElectricPotentialSignal.fromTimeKnots",
    summary: "Create a electric potential time history from continuous and jump knots.",
    description: concat!(
        "`fea.ElectricPotentialSignal.fromTimeKnots(id, knots, interpolation, extrapolation)`",
        " returns an `fea.ElectricPotentialSignal` whose value is a scalar in volts (V), ",
        "tabulated over time in seconds at the given knots.",
    ),
    keywords: &["fea", "signal", "time history", "electric potential", "knots", "SI units"],
    related: &["fea.SignalKnot.continuous", "fea.SignalKnot.jump", "fea.SignalInterpolation.linear", "fea.SignalExtrapolation.holdEndpoint", "fea.ElectricPotentialSignal.fromTimeSamples", "fea.ElectricalExcitation.terminalVoltage", "fea.ElectricalInitialCondition.potential", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                "The electric potential ordinate of every knot is a scalar in volts (V).",
                super::HISTORY_BEHAVIOR,
                super::HISTORY_SAMPLING,
                super::HISTORY_RESULT,
            ],
        ),
        section(
            "Use in a study",
            &[
                concat!(
                    "Signals of this class feed `fea.ElectricalExcitation.terminalVoltage`, ",
                    "`fea.ElectricalInitialCondition.potential`.",
                ),
                super::HISTORY_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "supply-ramp",
            "Ramp a terminal to 12 V in 1 ms",
            concat!(
                "knots = {fea.SignalKnot.continuous(0, 0), fea.SignalKnot.continuous(0.001, 12)};\n",
                "signal = fea.ElectricPotentialSignal.fromTimeKnots('supply_ramp', knots, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.holdEndpoint());",
            ),
            concat!(
                "assert(strcmp(class(signal), 'fea.ElectricPotentialSignal'));\n",
                "assert(signal.signal.electric_potential.signal_id == \"supply_ramp\");\n",
                "assert(signal.signal.electric_potential.shape == \"scalar\");\n",
                "assert(signal.signal.electric_potential.representation.tabulated.domain == \"time_seconds\");",
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
    FEA_ELECTRIC_POTENTIAL_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    name: "fea.ElectricPotentialSignal.fromTimeKnots",
    result: classes::ELECTRIC_POTENTIAL_SIGNAL,
    signature: concat!(
        "signal = fea.ElectricPotentialSignal.fromTimeKnots(id, knots, interpolation, ",
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
                " in volts (V).",
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
    output: ("signal", "Typed fea.ElectricPotentialSignal time history."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
