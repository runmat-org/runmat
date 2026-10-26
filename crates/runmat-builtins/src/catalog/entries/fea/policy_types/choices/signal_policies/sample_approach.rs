//! `fea.SignalSampleApproach` choices: how an initial condition reads one
//! value from a signal, including one-sided limits at jump knots.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod at {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "At an exact knot the sample is the knot's point value; for a jump knot that is the side selected by its `fea.SignalPointContinuity`. Between knots the sample is the interpolated value.",
                "`at` is the only approach allowed at an abscissa that is not a knot.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for ordinary initial conditions read from a continuous history, such as the starting temperature of a transient taken at time zero.",
            ],
        ),
        USE_SAMPLE_APPROACH,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "initial-temperature",
            "Read an initial temperature at time zero",
            "approach = fea.SignalSampleApproach.at();\nambient = fea.TemperatureSignal.constantKelvin(\"ambient\", 293.15);\ncondition = fea.ThermalInitialCondition.temperature(\"t0\", fea.InitialConditionTarget.wholeModel(), ...\n    ambient, 0, approach, \"heat_up\");",
            "assert(strcmp(class(approach), 'fea.SignalSampleApproach'));\nassert(strcmp(approach.approach, 'at'));\nassert(strcmp(class(condition), 'fea.ThermalInitialCondition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalSampleApproach.leftLimit",
        "fea.SignalSampleApproach.rightLimit",
        "fea.ThermalInitialCondition.temperature",
        "fea.StructuralInitialCondition.velocity",
        "fea.SignalKnot.jump",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalSampleApproach.at",
        summary: "Sample the value assigned at the abscissa itself.",
        description: "`fea.SignalSampleApproach.at()` returns the closed `fea.SignalSampleApproach` value `at`, which selects how an initial condition reads its signal at one abscissa.",
        keywords: &[
            "fea",
            "signal",
            "sample",
            "initial condition",
            "one-sided limit",
            "at",
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
        name: "fea.SignalSampleApproach.at",
        result: crate::fea_contract::policy_choices::classes::SIGNAL_SAMPLE_APPROACH,
        signature: "approach = fea.SignalSampleApproach.at()",
        inputs: [],
        output: ("approach", "Typed fea.SignalSampleApproach value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod left_limit {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The abscissa must be exactly a knot other than the first one. The sample is the knot's incoming value: the left limit of a jump knot, or the ordinary value of a continuous knot.",
                "A one-sided request between knots, or a left limit at the first knot, fails when the study samples the signal (`one-sided signal sampling requires an exact knot abscissa`).",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to start a step from the state just before a discontinuity, for example the part temperature immediately before an oven setpoint jump.",
            ],
        ),
        USE_SAMPLE_APPROACH,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "before-jump",
            "Start from the value just before a jump",
            "approach = fea.SignalSampleApproach.leftLimit();\nsignal = fea.TemperatureSignal.fromTimeKnots(\"oven\", {fea.SignalKnot.continuous(0, 293.15), ...\n    fea.SignalKnot.jump(60, 293.15, 473.15, fea.SignalPointContinuity.right()), ...\n    fea.SignalKnot.continuous(120, 473.15)}, fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());\ncondition = fea.ThermalInitialCondition.temperature(\"t_before\", fea.InitialConditionTarget.wholeModel(), ...\n    signal, 60, approach, \"soak\");",
            "assert(strcmp(approach.approach, 'left_limit'));\nassert(strcmp(class(condition), 'fea.ThermalInitialCondition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalSampleApproach.at",
        "fea.SignalSampleApproach.rightLimit",
        "fea.ThermalInitialCondition.temperature",
        "fea.StructuralInitialCondition.velocity",
        "fea.SignalKnot.jump",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalSampleApproach.leftLimit",
        summary: "Sample the limit approaching a knot from below.",
        description: "`fea.SignalSampleApproach.leftLimit()` returns the closed `fea.SignalSampleApproach` value `left_limit`, which selects how an initial condition reads its signal at one abscissa.",
        keywords: &[
            "fea",
            "signal",
            "sample",
            "initial condition",
            "one-sided limit",
            "leftLimit",
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
        name: "fea.SignalSampleApproach.leftLimit",
        result: crate::fea_contract::policy_choices::classes::SIGNAL_SAMPLE_APPROACH,
        signature: "approach = fea.SignalSampleApproach.leftLimit()",
        inputs: [],
        output: ("approach", "Typed fea.SignalSampleApproach value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod right_limit {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The abscissa must be exactly a knot other than the last one. The sample is the knot's outgoing value: the right limit of a jump knot, or the ordinary value of a continuous knot.",
                "A one-sided request between knots, or a right limit at the last knot, fails when the study samples the signal.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to start a step from the state just after a discontinuity, for example the new setpoint applied at the jump time.",
            ],
        ),
        USE_SAMPLE_APPROACH,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "after-jump",
            "Start from the value just after a jump",
            "approach = fea.SignalSampleApproach.rightLimit();\nsignal = fea.TemperatureSignal.fromTimeKnots(\"oven\", {fea.SignalKnot.continuous(0, 293.15), ...\n    fea.SignalKnot.jump(60, 293.15, 473.15, fea.SignalPointContinuity.right()), ...\n    fea.SignalKnot.continuous(120, 473.15)}, fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());\ncondition = fea.ThermalInitialCondition.temperature(\"t_after\", fea.InitialConditionTarget.wholeModel(), ...\n    signal, 60, approach, \"soak\");",
            "assert(strcmp(approach.approach, 'right_limit'));\nassert(strcmp(class(condition), 'fea.ThermalInitialCondition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalSampleApproach.at",
        "fea.SignalSampleApproach.leftLimit",
        "fea.ThermalInitialCondition.temperature",
        "fea.StructuralInitialCondition.velocity",
        "fea.SignalKnot.jump",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalSampleApproach.rightLimit",
        summary: "Sample the limit approaching a knot from above.",
        description: "`fea.SignalSampleApproach.rightLimit()` returns the closed `fea.SignalSampleApproach` value `right_limit`, which selects how an initial condition reads its signal at one abscissa.",
        keywords: &[
            "fea",
            "signal",
            "sample",
            "initial condition",
            "one-sided limit",
            "rightLimit",
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
        name: "fea.SignalSampleApproach.rightLimit",
        result: crate::fea_contract::policy_choices::classes::SIGNAL_SAMPLE_APPROACH,
        signature: "approach = fea.SignalSampleApproach.rightLimit()",
        inputs: [],
        output: ("approach", "Typed fea.SignalSampleApproach value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &at::CATALOG_ENTRY,
    &left_limit::CATALOG_ENTRY,
    &right_limit::CATALOG_ENTRY,
];
