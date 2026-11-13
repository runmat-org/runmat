//! `fea.SignalPointContinuity` choices: which side of a jump knot is its
//! point value.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod left {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The signal is left-continuous at the jump: the value exactly at the abscissa equals the value approached from earlier times, and the new value takes effect immediately after the knot.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when an event takes effect just after its time stamp, for example a load removed at `t` that is still fully applied at `t` itself.",
            ],
        ),
        USE_CONTINUITY,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "release",
            "Release a load just after its time stamp",
            "continuity = fea.SignalPointContinuity.left();\nknot = fea.SignalKnot.jump(0.5, 1200, 0, continuity);",
            "assert(strcmp(class(continuity), 'fea.SignalPointContinuity'));\nassert(strcmp(continuity.continuity, 'left'));\nassert(strcmp(class(knot), 'fea.SignalKnot'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalPointContinuity.right",
        "fea.SignalKnot.jump",
        "fea.SignalKnot.continuous",
        "fea.TemperatureSignal.fromTimeKnots",
        "fea.SignalSampleApproach.at",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalPointContinuity.left",
        summary: "Assign a jump knot's point value to its left-hand limit.",
        description: "`fea.SignalPointContinuity.left()` returns the closed `fea.SignalPointContinuity` value `left`, which fixes the value a jump knot takes exactly at its abscissa.",
        keywords: &[
            "fea",
            "signal",
            "jump",
            "knot",
            "continuity",
            "left",
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
        name: "fea.SignalPointContinuity.left",
        result: crate::fea_contract::policy_types::classes::SIGNAL_POINT_CONTINUITY,
        signature: "continuity = fea.SignalPointContinuity.left()",
        inputs: [],
        output: ("continuity", "Typed fea.SignalPointContinuity value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod right {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The signal is right-continuous at the jump: the value exactly at the abscissa equals the value that applies just after it. This matches the usual convention for a step applied at time `t`.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for step loads and setpoint changes that are in effect from their time stamp onward.",
            ],
        ),
        USE_CONTINUITY,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "step-on",
            "Apply a step heat flux from its time stamp",
            "continuity = fea.SignalPointContinuity.right();\nknot = fea.SignalKnot.jump(2.0, 0, 1500, continuity);",
            "assert(strcmp(class(continuity), 'fea.SignalPointContinuity'));\nassert(strcmp(continuity.continuity, 'right'));\nassert(strcmp(class(knot), 'fea.SignalKnot'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SignalPointContinuity.left",
        "fea.SignalKnot.jump",
        "fea.SignalKnot.continuous",
        "fea.TemperatureSignal.fromTimeKnots",
        "fea.SignalSampleApproach.at",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalPointContinuity.right",
        summary: "Assign a jump knot's point value to its right-hand limit.",
        description: "`fea.SignalPointContinuity.right()` returns the closed `fea.SignalPointContinuity` value `right`, which fixes the value a jump knot takes exactly at its abscissa.",
        keywords: &[
            "fea",
            "signal",
            "jump",
            "knot",
            "continuity",
            "right",
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
        name: "fea.SignalPointContinuity.right",
        result: crate::fea_contract::policy_types::classes::SIGNAL_POINT_CONTINUITY,
        signature: "continuity = fea.SignalPointContinuity.right()",
        inputs: [],
        output: ("continuity", "Typed fea.SignalPointContinuity value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&left::CATALOG_ENTRY, &right::CATALOG_ENTRY];
