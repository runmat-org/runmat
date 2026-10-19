use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalKnot.continuous",
    summary: "Create a knot where a time history is continuous.",
    description: concat!(
        "`fea.SignalKnot.continuous(abscissa, value)` returns an `fea.SignalKnot` at time ",
        "`abscissa` whose value is `value` from the left, from the right, and at the point ",
        "itself. The `knot` property holds `abscissa` and a `value` struct with a ",
        "`Continuous` ordinate.",
    ),
    keywords: &["fea", "signal", "knot", "continuous", "time history"],
    related: &["fea.SignalKnot.jump", "fea.ForceSignal.fromTimeKnots", "fea.PressureSignal.fromTimeKnots", "fea.SignalInterpolation.linear"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`abscissa` must be a finite real scalar. `value` must be a finite real scalar or a ",
                    "vector with exactly three components; other sizes, logical, text, complex, and ",
                    "nonfinite values fail with `RunMat:fea:InvalidInput`.",
                ),
                super::KNOT_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::KNOT_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "ramp-start",
            "Start a force ramp at rest",
            concat!(
                "start = fea.SignalKnot.continuous(0, [0 0 0]);\n",
                "peak = fea.SignalKnot.continuous(0.2, [0 0 -2500]);",
            ),
            concat!(
                "assert(strcmp(class(start), 'fea.SignalKnot'));\n",
                "assert(peak.knot.abscissa == 0.2);\n",
                "assert(numel(peak.knot.value.Continuous) == 3);\n",
                "assert(peak.knot.value.Continuous(3) == -2500);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Are knot times always seconds?",
            concat!(
                "Yes. Knots feed time-domain histories (`fromTimeKnots`), whose abscissa is time in ",
                "seconds.",
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
    ],
    notes: &[],
});

policy_entry!(
    FEA_SIGNAL_KNOT_CONTINUOUS_CATALOG_ENTRY,
    name: "fea.SignalKnot.continuous",
    result: classes::SIGNAL_KNOT,
    signature: "knot = fea.SignalKnot.continuous(abscissa, value)",
    inputs: [
        (
            "abscissa",
            NumericScalar,
            "Finite time of the knot in seconds.",
            PolicyArg::RealScalar,
        ),
        (
            "value",
            NumericArray,
            concat!(
                "Finite scalar, or three-component row or column vector, in the owning signal's SI ",
                "unit.",
            ),
            PolicyArg::ScalarOrVector3,
        ),
    ],
    output: ("knot", "Typed fea.SignalKnot."),
    integer_capabilities: &integer::CONTINUOUS_KNOT_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
