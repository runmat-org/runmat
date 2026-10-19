//! `fea.SpectrumExtrapolation` choices: modes outside the spectrum axis.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod reject {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "If a retained mode's natural frequency falls below the first or above the last axis point, the response-spectrum solve fails instead of inventing an ordinate.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to guarantee that every contributing mode is covered by the supplied spectrum; extend the axis or adjust the retained modes when it fails.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "reject",
            "Select reject for a response spectrum",
            "extrapolation = fea.SpectrumExtrapolation.reject();",
            "assert(strcmp(class(extrapolation), 'fea.SpectrumExtrapolation'));\nassert(strcmp(extrapolation.value, 'reject'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumExtrapolation.holdEndpoint",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumExtrapolation.reject",
        summary: "Fail when a mode lies outside the spectrum axis.",
        description: "`fea.SpectrumExtrapolation.reject()` returns the closed `fea.SpectrumExtrapolation` value `reject` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumExtrapolation",
            "reject",
            "modal",
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
        name: "fea.SpectrumExtrapolation.reject",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_EXTRAPOLATION,
        signature: "extrapolation = fea.SpectrumExtrapolation.reject()",
        inputs: [],
        output: ("extrapolation", "Typed fea.SpectrumExtrapolation value."),
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
                "A mode below the first axis point uses the first ordinate; a mode above the last point uses the last ordinate.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when the specification states that the spectrum stays flat beyond its tabulated range, for example a constant zero-period acceleration plateau at high frequency.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "hold-endpoint",
            "Select holdEndpoint for a response spectrum",
            "extrapolation = fea.SpectrumExtrapolation.holdEndpoint();",
            "assert(strcmp(class(extrapolation), 'fea.SpectrumExtrapolation'));\nassert(strcmp(extrapolation.value, 'hold_endpoint'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumExtrapolation.reject",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumExtrapolation.holdEndpoint",
        summary: "Use the nearest endpoint ordinate outside the spectrum axis.",
        description: "`fea.SpectrumExtrapolation.holdEndpoint()` returns the closed `fea.SpectrumExtrapolation` value `hold_endpoint` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumExtrapolation",
            "holdEndpoint",
            "modal",
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
        name: "fea.SpectrumExtrapolation.holdEndpoint",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_EXTRAPOLATION,
        signature: "extrapolation = fea.SpectrumExtrapolation.holdEndpoint()",
        inputs: [],
        output: ("extrapolation", "Typed fea.SpectrumExtrapolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&reject::CATALOG_ENTRY, &hold_endpoint::CATALOG_ENTRY];
