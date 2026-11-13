use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.HeatFluxSignal.fromTimeKnots",
    summary: "Create a heat flux time history from continuous and jump knots.",
    description: concat!(
        "`fea.HeatFluxSignal.fromTimeKnots(id, knots, interpolation, extrapolation)` returns ",
        "an `fea.HeatFluxSignal` whose value is a scalar in watts per square meter (W/m^2), ",
        "tabulated over time in seconds at the given knots.",
    ),
    keywords: &["fea", "signal", "time history", "heat flux", "knots", "SI units"],
    related: &["fea.SignalKnot.continuous", "fea.SignalKnot.jump", "fea.SignalInterpolation.linear", "fea.SignalExtrapolation.holdEndpoint", "fea.HeatFluxSignal.fromTimeSamples", "fea.ThermalExcitation.heatFlux", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                "The heat flux ordinate of every knot is a scalar in watts per square meter (W/m^2).",
                super::HISTORY_BEHAVIOR,
                super::HISTORY_SAMPLING,
                super::HISTORY_RESULT,
            ],
        ),
        section(
            "Use in a study",
            &[
                "Signals of this class feed `fea.ThermalExcitation.heatFlux`.",
                super::HISTORY_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "heater",
            "Switch on a 5 kW/m^2 heat flux at 10 s",
            concat!(
                "knots = {fea.SignalKnot.continuous(0, 0), fea.SignalKnot.jump(10, 0, 5000, fea.SignalPointContinuity.right()), fea.SignalKnot.continuous(120, 5000)};\n",
                "signal = fea.HeatFluxSignal.fromTimeKnots('heater', knots, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.holdEndpoint());",
            ),
            concat!(
                "assert(strcmp(class(signal), 'fea.HeatFluxSignal'));\n",
                "assert(signal.signal.heat_flux.signal_id == \"heater\");\n",
                "assert(signal.signal.heat_flux.shape == \"scalar\");\n",
                "assert(signal.signal.heat_flux.representation.tabulated.domain == \"time_seconds\");",
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
    FEA_HEAT_FLUX_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    name: "fea.HeatFluxSignal.fromTimeKnots",
    result: classes::HEAT_FLUX_SIGNAL,
    signature: "signal = fea.HeatFluxSignal.fromTimeKnots(id, knots, interpolation, extrapolation)",
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
                " in watts per square meter (W/m^2).",
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
    output: ("signal", "Typed fea.HeatFluxSignal time history."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
