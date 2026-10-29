use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RotationSignal.fromTimeKnots",
    summary: "Create a rotation time history from continuous and jump knots.",
    description: concat!(
        "`fea.RotationSignal.fromTimeKnots(id, knots, interpolation, extrapolation)` returns ",
        "an `fea.RotationSignal` whose value is a three-component vector in the global frame ",
        "in radians (rad), as a rotation vector, tabulated over time in seconds at the given ",
        "knots.",
    ),
    keywords: &["fea", "signal", "time history", "rotation", "knots", "SI units"],
    related: &["fea.SignalKnot.continuous", "fea.SignalKnot.jump", "fea.SignalInterpolation.linear", "fea.SignalExtrapolation.holdEndpoint", "fea.RotationSignal.fromTimeSamples", "fea.model"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "The rotation ordinate of every knot is a three-component vector in radians (rad), as",
                    " a rotation vector. Vector signals are expressed in the global coordinate frame.",
                ),
                super::HISTORY_BEHAVIOR,
                super::HISTORY_SAMPLING,
                super::HISTORY_RESULT,
            ],
        ),
        section(
            "Use in a study",
            &[
                "Use the signal wherever a model object consumes a rotation signal.",
                super::HISTORY_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "twist",
            "Rotate 0.01 rad about z over 1 s",
            concat!(
                "knots = {fea.SignalKnot.continuous(0, [0 0 0]), fea.SignalKnot.continuous(1, [0 0 0.01])};\n",
                "signal = fea.RotationSignal.fromTimeKnots('twist', knots, ...\n",
                "    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.holdEndpoint());",
            ),
            concat!(
                "assert(strcmp(class(signal), 'fea.RotationSignal'));\n",
                "assert(signal.signal.rotation.signal_id == \"twist\");\n",
                "assert(signal.signal.rotation.shape == \"vector3\");\n",
                "assert(signal.signal.rotation.representation.tabulated.domain == \"time_seconds\");",
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
    FEA_ROTATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    name: "fea.RotationSignal.fromTimeKnots",
    result: classes::ROTATION_SIGNAL,
    signature: "signal = fea.RotationSignal.fromTimeKnots(id, knots, interpolation, extrapolation)",
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
                "Row or column cell array of at least two fea.SignalKnot values with three-component ",
                "ordinates in radians (rad), as a rotation vector.",
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
    output: ("signal", "Typed fea.RotationSignal time history."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
