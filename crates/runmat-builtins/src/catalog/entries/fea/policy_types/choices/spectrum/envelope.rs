//! `fea.SpectrumEnvelope` choices: which response peaks the ordinates bound.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod positive {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The ordinates are the largest positive oscillator responses. Each mode's generalized peak keeps the sign of its participation factor times the ordinate.",
                "Combination rules work on magnitudes or squares, so the sign matters only where responses are summed with sign, such as the rigid-response sum and a missing-mass correction.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when the specification provides one-sided positive spectra.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "positive",
            "Select positive for a response spectrum",
            "envelope = fea.SpectrumEnvelope.positive();",
            "assert(strcmp(class(envelope), 'fea.SpectrumEnvelope'));\nassert(strcmp(envelope.value, 'positive'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumEnvelope.negative",
        "fea.SpectrumEnvelope.maximax",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumEnvelope.positive",
        summary: "Use ordinates that are peaks of positive response.",
        description: "`fea.SpectrumEnvelope.positive()` returns the closed `fea.SpectrumEnvelope` value `positive` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumEnvelope",
            "positive",
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
        name: "fea.SpectrumEnvelope.positive",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_ENVELOPE,
        signature: "envelope = fea.SpectrumEnvelope.positive()",
        inputs: [],
        output: ("envelope", "Typed fea.SpectrumEnvelope value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod negative {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The ordinates are magnitudes of the most negative oscillator responses. RunMat reverses the sign of each mode's generalized peak (and of the residual-mode and missing-mass contributions) so signed sums point the right way.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when the negative-going spectrum of a one-sided shock governs and is supplied separately from the positive one.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "negative",
            "Select negative for a response spectrum",
            "envelope = fea.SpectrumEnvelope.negative();",
            "assert(strcmp(class(envelope), 'fea.SpectrumEnvelope'));\nassert(strcmp(envelope.value, 'negative'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumEnvelope.positive",
        "fea.SpectrumEnvelope.maximax",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumEnvelope.negative",
        summary: "Use ordinates that are peaks of negative response.",
        description: "`fea.SpectrumEnvelope.negative()` returns the closed `fea.SpectrumEnvelope` value `negative` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumEnvelope",
            "negative",
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
        name: "fea.SpectrumEnvelope.negative",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_ENVELOPE,
        signature: "envelope = fea.SpectrumEnvelope.negative()",
        inputs: [],
        output: ("envelope", "Typed fea.SpectrumEnvelope value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod maximax {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The ordinates are the maximum absolute oscillator response over the whole event (maximax). RunMat treats the generalized peaks with positive sign; the combined result is a statistical magnitude envelope, not a signed response.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for standard shock response spectra and equipment qualification spectra, which are almost always maximax.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "maximax",
            "Select maximax for a response spectrum",
            "envelope = fea.SpectrumEnvelope.maximax();",
            "assert(strcmp(class(envelope), 'fea.SpectrumEnvelope'));\nassert(strcmp(envelope.value, 'maximax'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumEnvelope.positive",
        "fea.SpectrumEnvelope.negative",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumEnvelope.maximax",
        summary: "Use ordinates that are the larger of positive and negative peaks.",
        description: "`fea.SpectrumEnvelope.maximax()` returns the closed `fea.SpectrumEnvelope` value `maximax` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumEnvelope",
            "maximax",
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
        name: "fea.SpectrumEnvelope.maximax",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_ENVELOPE,
        signature: "envelope = fea.SpectrumEnvelope.maximax()",
        inputs: [],
        output: ("envelope", "Typed fea.SpectrumEnvelope value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &positive::CATALOG_ENTRY,
    &negative::CATALOG_ENTRY,
    &maximax::CATALOG_ENTRY,
];
