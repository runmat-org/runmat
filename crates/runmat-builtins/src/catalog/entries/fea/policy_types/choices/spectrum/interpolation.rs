//! `fea.SpectrumInterpolation` choices: how ordinates are read at a mode's frequency.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod linear_frequency {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each mode's ordinate is interpolated linearly in hertz between the two neighboring axis points.",
                "The spectrum axis must be a frequency axis (`fea.SpectrumAxis.frequencyHertz`); `fea.ResponseSpectrum.new` rejects a period axis with this choice.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for densely sampled spectra tabulated in frequency.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "linear-frequency",
            "Select linearFrequency for a response spectrum",
            "interpolation = fea.SpectrumInterpolation.linearFrequency();",
            "assert(strcmp(class(interpolation), 'fea.SpectrumInterpolation'));\nassert(strcmp(interpolation.value, 'linear_frequency'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumInterpolation.linearPeriod",
        "fea.SpectrumInterpolation.logLog",
        "fea.SpectrumAxis.frequencyHertz",
        "fea.SpectrumAxis.periodSeconds",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumInterpolation.linearFrequency",
        summary: "Interpolate ordinates linearly in frequency.",
        description: "`fea.SpectrumInterpolation.linearFrequency()` returns the closed `fea.SpectrumInterpolation` value `linear_frequency` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumInterpolation",
            "linearFrequency",
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
        name: "fea.SpectrumInterpolation.linearFrequency",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_INTERPOLATION,
        signature: "interpolation = fea.SpectrumInterpolation.linearFrequency()",
        inputs: [],
        output: ("interpolation", "Typed fea.SpectrumInterpolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod linear_period {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each mode's ordinate is interpolated linearly in seconds of period between neighboring axis points.",
                "The axis must be a period axis (`fea.SpectrumAxis.periodSeconds`).",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for seismic design spectra published against period.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "linear-period",
            "Select linearPeriod for a response spectrum",
            "interpolation = fea.SpectrumInterpolation.linearPeriod();",
            "assert(strcmp(class(interpolation), 'fea.SpectrumInterpolation'));\nassert(strcmp(interpolation.value, 'linear_period'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumInterpolation.linearFrequency",
        "fea.SpectrumInterpolation.logLog",
        "fea.SpectrumAxis.frequencyHertz",
        "fea.SpectrumAxis.periodSeconds",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumInterpolation.linearPeriod",
        summary: "Interpolate ordinates linearly in period.",
        description: "`fea.SpectrumInterpolation.linearPeriod()` returns the closed `fea.SpectrumInterpolation` value `linear_period` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumInterpolation",
            "linearPeriod",
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
        name: "fea.SpectrumInterpolation.linearPeriod",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_INTERPOLATION,
        signature: "interpolation = fea.SpectrumInterpolation.linearPeriod()",
        inputs: [],
        output: ("interpolation", "Typed fea.SpectrumInterpolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod log_log {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Between neighboring points, the logarithm of the ordinate varies linearly with the logarithm of the abscissa, so constant dB-per-octave ramps are reproduced exactly. It works with either a frequency or a period axis.",
                "Every ordinate must be strictly positive; `fea.ResponseSpectrum.new` rejects zero ordinates with this choice.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for qualification spectra specified by breakpoints and slopes, such as shock requirements in dB per octave.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "log-log",
            "Select logLog for a response spectrum",
            "interpolation = fea.SpectrumInterpolation.logLog();",
            "assert(strcmp(class(interpolation), 'fea.SpectrumInterpolation'));\nassert(strcmp(interpolation.value, 'log_log'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SpectrumInterpolation.linearFrequency",
        "fea.SpectrumInterpolation.linearPeriod",
        "fea.SpectrumAxis.frequencyHertz",
        "fea.SpectrumAxis.periodSeconds",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumInterpolation.logLog",
        summary: "Interpolate ordinates as straight lines on log-log axes.",
        description: "`fea.SpectrumInterpolation.logLog()` returns the closed `fea.SpectrumInterpolation` value `log_log` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumInterpolation",
            "logLog",
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
        name: "fea.SpectrumInterpolation.logLog",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_INTERPOLATION,
        signature: "interpolation = fea.SpectrumInterpolation.logLog()",
        inputs: [],
        output: ("interpolation", "Typed fea.SpectrumInterpolation value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &linear_frequency::CATALOG_ENTRY,
    &linear_period::CATALOG_ENTRY,
    &log_log::CATALOG_ENTRY,
];
