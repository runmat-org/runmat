//! `fea.SignalExtrapolation` choices: how a signal answers samples outside
//! its first and last knot.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod reject {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A sample requested before the first knot or after the last knot fails the realization that asked for it with `sample lies outside signal domain`. RunMat never invents data beyond the table.",
                "The check happens when a step samples the signal, so a transient step that runs longer than the history fails before it publishes results.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it as the safe default: it is the default extrapolation of `fromFunction` signals and is required by `fea.StructuralMotion.acceleration` support motion. Extend the table to cover every time the study will sample.",
            ],
        ),
        USE_TABULATED,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "bounded-history",
            "Bound an acceleration history to its recorded window",
            "policy = fea.SignalExtrapolation.reject();\nbase = fea.AccelerationSignal.fromTimeSamples(\"base_shake\", [0 0.01 0.02 0.03], ...\n    [0 0 0; 0 0 4.9; 0 0 -4.9; 0 0 0], fea.SignalInterpolation.linear(), policy);",
            "assert(strcmp(class(policy), 'fea.SignalExtrapolation'));\nassert(strcmp(policy.extrapolation, 'reject'));\nassert(strcmp(base.signal.acceleration.representation.tabulated.extrapolation, 'reject'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalExtrapolation.holdEndpoint",
        "fea.SignalExtrapolation.zero",
        "fea.SignalInterpolation.linear",
        "fea.PressureSignal.fromTimeSamples",
        "fea.TemperatureSignal.fromTimeSamples",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalExtrapolation.reject",
        summary: "Reject any sample outside the signal's knot range.",
        description: "`fea.SignalExtrapolation.reject()` returns the closed `fea.SignalExtrapolation` value `reject`. It takes no arguments and governs samples requested before the first or after the last knot.",
        keywords: &[
            "fea",
            "signal",
            "extrapolation",
            "reject",
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
        name: "fea.SignalExtrapolation.reject",
        result: crate::fea_contract::policy_types::classes::SIGNAL_EXTRAPOLATION,
        signature: "policy = fea.SignalExtrapolation.reject()",
        inputs: [],
        output: ("policy", "Typed fea.SignalExtrapolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod hold_endpoint {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Before the first knot the signal equals the first knot's point value; after the last knot it equals the last knot's point value. Inside the range the interpolation policy applies unchanged.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for a ramp that settles to a steady value, for example a pressure that rises to its operating level and then stays there for the rest of a longer transient.",
            ],
        ),
        USE_TABULATED,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "settle",
            "Ramp a pressure and hold it",
            "policy = fea.SignalExtrapolation.holdEndpoint();\nramp = fea.PressureSignal.fromTimeSamples(\"cabin\", [0 0.5 1.0], [0 4e4 6e4], ...\n    fea.SignalInterpolation.linear(), policy);",
            "assert(strcmp(class(policy), 'fea.SignalExtrapolation'));\nassert(strcmp(policy.extrapolation, 'hold_endpoint'));\nassert(strcmp(ramp.signal.pressure.representation.tabulated.extrapolation, 'hold_endpoint'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalExtrapolation.reject",
        "fea.SignalExtrapolation.zero",
        "fea.SignalInterpolation.linear",
        "fea.PressureSignal.fromTimeSamples",
        "fea.TemperatureSignal.fromTimeSamples",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalExtrapolation.holdEndpoint",
        summary: "Hold the nearest endpoint value outside the knot range.",
        description: "`fea.SignalExtrapolation.holdEndpoint()` returns the closed `fea.SignalExtrapolation` value `hold_endpoint`. It takes no arguments and governs samples requested before the first or after the last knot.",
        keywords: &[
            "fea",
            "signal",
            "extrapolation",
            "holdEndpoint",
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
        name: "fea.SignalExtrapolation.holdEndpoint",
        result: crate::fea_contract::policy_types::classes::SIGNAL_EXTRAPOLATION,
        signature: "policy = fea.SignalExtrapolation.holdEndpoint()",
        inputs: [],
        output: ("policy", "Typed fea.SignalExtrapolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod zero {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Before the first knot and after the last knot every component of the signal is zero in its canonical SI unit. Inside the range the interpolation policy applies unchanged.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for pulses and transient loads that are physically absent outside the tabulated window, such as an impact force or a heat-flux burst.",
            ],
        ),
        USE_TABULATED,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "pulse",
            "Model a finite force pulse",
            "policy = fea.SignalExtrapolation.zero();\npulse = fea.ForceSignal.fromTimeSamples(\"impact\", [0.010 0.012 0.014 0.016], ...\n    [0 0 0; 0 0 -2000; 0 0 -2000; 0 0 0], fea.SignalInterpolation.linear(), policy);",
            "assert(strcmp(class(policy), 'fea.SignalExtrapolation'));\nassert(strcmp(policy.extrapolation, 'zero'));\nassert(strcmp(pulse.signal.force.representation.tabulated.extrapolation, 'zero'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Is zero extrapolation safe for absolute temperature?",
            "Rarely. Zero means 0 K for a `fea.TemperatureSignal`. Use `holdEndpoint`, or a `fea.TemperatureDifferenceSignal` when a zero offset is intended.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.SignalExtrapolation.reject",
        "fea.SignalExtrapolation.holdEndpoint",
        "fea.SignalInterpolation.linear",
        "fea.PressureSignal.fromTimeSamples",
        "fea.TemperatureSignal.fromTimeSamples",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalExtrapolation.zero",
        summary: "Use zero outside the knot range.",
        description: "`fea.SignalExtrapolation.zero()` returns the closed `fea.SignalExtrapolation` value `zero`. It takes no arguments and governs samples requested before the first or after the last knot.",
        keywords: &[
            "fea",
            "signal",
            "extrapolation",
            "zero",
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
        name: "fea.SignalExtrapolation.zero",
        result: crate::fea_contract::policy_types::classes::SIGNAL_EXTRAPOLATION,
        signature: "policy = fea.SignalExtrapolation.zero()",
        inputs: [],
        output: ("policy", "Typed fea.SignalExtrapolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &reject::CATALOG_ENTRY,
    &hold_endpoint::CATALOG_ENTRY,
    &zero::CATALOG_ENTRY,
];
