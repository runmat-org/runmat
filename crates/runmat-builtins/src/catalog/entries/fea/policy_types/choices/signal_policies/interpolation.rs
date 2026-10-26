//! `fea.SignalInterpolation` choices: how a tabulated or generated signal is
//! evaluated strictly between knots.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod left_endpoint {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Between two knots `t(i) < t < t(i+1)` the signal equals the outgoing value of the left knot, so a tabulated history becomes a staircase that changes value only at knots. For a jump knot the outgoing value is its right limit.",
                "Exactly at a knot, an `at` sample still reads the knot's own point value; interpolation applies only strictly between knots.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for piecewise-constant programs: load steps held between switching times, set points from a digital controller, or duty cycles recorded as the value that starts each interval.",
            ],
        ),
        USE_TABULATED,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "load-history",
            "Use leftEndpoint interpolation for a force history",
            "policy = fea.SignalInterpolation.leftEndpoint();\nload = fea.ForceSignal.fromTimeSamples(\"tip_load\", [0 0.1 0.2 0.3], ...\n    [0 0 0; 0 0 -500; 0 0 -500; 0 0 0], policy, fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(policy), 'fea.SignalInterpolation'));\nassert(strcmp(policy.interpolation, 'left_endpoint'));\nassert(strcmp(load.signal.force.representation.tabulated.interpolation, 'left_endpoint'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How is this different from rightEndpoint?",
            "Both produce steps. `leftEndpoint` uses the value at the start of each interval; `rightEndpoint` uses the value arriving at the end of it, so the step happens one interval earlier.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.SignalInterpolation.rightEndpoint",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
        "fea.ForceSignal.fromTimeSamples",
        "fea.ForceSignal.fromTimeKnots",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalInterpolation.leftEndpoint",
        summary: "Hold each interval at the value leaving its left knot (zero-order hold).",
        description: "`fea.SignalInterpolation.leftEndpoint()` returns the closed `fea.SignalInterpolation` value `left_endpoint`. It takes no arguments and is stored by the signal that receives it.",
        keywords: &[
            "fea",
            "signal",
            "interpolation",
            "leftEndpoint",
            "time history",
            "policy",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.SignalInterpolation.leftEndpoint",
        result: crate::fea_contract::policy_types::classes::SIGNAL_INTERPOLATION,
        signature: "policy = fea.SignalInterpolation.leftEndpoint()",
        inputs: [],
        output: ("policy", "Typed fea.SignalInterpolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod right_endpoint {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Between two knots `t(i) < t < t(i+1)` the signal equals the incoming value of the right knot. For a jump knot the incoming value is its left limit. The result is a staircase that takes each new value immediately after the previous knot.",
                "Exactly at a knot, an `at` sample reads the knot's point value.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when each tabulated value describes the interval that ends at its time stamp, for example a logger that records an average or a commanded value at the end of every period.",
            ],
        ),
        USE_TABULATED,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "load-history",
            "Use rightEndpoint interpolation for a force history",
            "policy = fea.SignalInterpolation.rightEndpoint();\nload = fea.ForceSignal.fromTimeSamples(\"tip_load\", [0 0.1 0.2 0.3], ...\n    [0 0 0; 0 0 -500; 0 0 -500; 0 0 0], policy, fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(policy), 'fea.SignalInterpolation'));\nassert(strcmp(policy.interpolation, 'right_endpoint'));\nassert(strcmp(load.signal.force.representation.tabulated.interpolation, 'right_endpoint'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalInterpolation.leftEndpoint",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
        "fea.ForceSignal.fromTimeSamples",
        "fea.ForceSignal.fromTimeKnots",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalInterpolation.rightEndpoint",
        summary: "Hold each interval at the value arriving at its right knot.",
        description: "`fea.SignalInterpolation.rightEndpoint()` returns the closed `fea.SignalInterpolation` value `right_endpoint`. It takes no arguments and is stored by the signal that receives it.",
        keywords: &[
            "fea",
            "signal",
            "interpolation",
            "rightEndpoint",
            "time history",
            "policy",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.SignalInterpolation.rightEndpoint",
        result: crate::fea_contract::policy_types::classes::SIGNAL_INTERPOLATION,
        signature: "policy = fea.SignalInterpolation.rightEndpoint()",
        inputs: [],
        output: ("policy", "Typed fea.SignalInterpolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod linear {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Between two knots the value is `(1 - w)*v(i) + w*v(i+1)` with `w = (t - t(i)) / (t(i+1) - t(i))`, using the outgoing value of the left knot and the incoming value of the right knot. Vector signals interpolate each global component independently.",
                "This is the default interpolation of `fromFunction` signals, so a generated signal behaves like the piecewise-linear table built from its samples.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for physically continuous histories such as ramps, measured accelerations, and temperature transients. Support motion (`fea.StructuralMotion.acceleration`) requires it.",
            ],
        ),
        USE_TABULATED,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "load-history",
            "Use linear interpolation for a force history",
            "policy = fea.SignalInterpolation.linear();\nload = fea.ForceSignal.fromTimeSamples(\"tip_load\", [0 0.1 0.2 0.3], ...\n    [0 0 0; 0 0 -500; 0 0 -500; 0 0 0], policy, fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(policy), 'fea.SignalInterpolation'));\nassert(strcmp(policy.interpolation, 'linear'));\nassert(strcmp(load.signal.force.representation.tabulated.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Does linear interpolation smooth a jump knot?",
            "No. The interval to the left of a jump ends at its left limit and the interval to the right starts at its right limit, so the discontinuity is preserved.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.SignalInterpolation.leftEndpoint",
        "fea.SignalInterpolation.rightEndpoint",
        "fea.SignalExtrapolation.reject",
        "fea.ForceSignal.fromTimeSamples",
        "fea.ForceSignal.fromTimeKnots",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalInterpolation.linear",
        summary: "Interpolate linearly between neighboring knots.",
        description: "`fea.SignalInterpolation.linear()` returns the closed `fea.SignalInterpolation` value `linear`. It takes no arguments and is stored by the signal that receives it.",
        keywords: &[
            "fea",
            "signal",
            "interpolation",
            "linear",
            "time history",
            "policy",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.SignalInterpolation.linear",
        result: crate::fea_contract::policy_types::classes::SIGNAL_INTERPOLATION,
        signature: "policy = fea.SignalInterpolation.linear()",
        inputs: [],
        output: ("policy", "Typed fea.SignalInterpolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &left_endpoint::CATALOG_ENTRY,
    &right_endpoint::CATALOG_ENTRY,
    &linear::CATALOG_ENTRY,
];
