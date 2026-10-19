//! `fea.SpectrumResponse` choices for relative response quantities.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod relative_displacement {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each ordinate is the peak displacement of a damped single-degree-of-freedom oscillator relative to its moving base, in meters, at the axis frequency or period.",
                "A mode's generalized peak is its directional participation factor times the interpolated ordinate; that peak is already a modal displacement, so deformation, strain, and stress fields are recovered directly.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for displacement design spectra and for checking clearances and relative motion.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "relative-displacement",
            "Select relativeDisplacement for a response spectrum",
            "response = fea.SpectrumResponse.relativeDisplacement();",
            "assert(strcmp(class(response), 'fea.SpectrumResponse'));\nassert(strcmp(response.value, 'relative_displacement'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which spectrum types give stress and displacement fields?",
            "Relative displacement, pseudo-velocity, and pseudo-acceleration spectra. RunMat converts their modal peaks to modal displacement (directly, divided by omega, or divided by omega squared) and recovers deformation fields; the other response quantities are combined but not converted.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.SpectrumResponse.relativeVelocity",
        "fea.SpectrumResponse.relativeAcceleration",
        "fea.SpectrumResponse.absoluteAcceleration",
        "fea.SpectrumResponse.pseudoVelocity",
        "fea.SpectrumResponse.pseudoAcceleration",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumResponse.relativeDisplacement",
        summary: "Read spectrum ordinates as peak relative displacement in meters.",
        description: "`fea.SpectrumResponse.relativeDisplacement()` returns the closed `fea.SpectrumResponse` value `relative_displacement` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumResponse",
            "relativeDisplacement",
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
        name: "fea.SpectrumResponse.relativeDisplacement",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_RESPONSE,
        signature: "response = fea.SpectrumResponse.relativeDisplacement()",
        inputs: [],
        output: ("response", "Typed fea.SpectrumResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod relative_velocity {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each ordinate is the peak velocity of a damped oscillator relative to its base, in m/s.",
                "RunMat combines the modal peaks of this quantity, but it does not convert them to displacement: displacement-derived fields (deformation, strain, stress) are reported as unavailable for this response, and the result says so instead of guessing.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when the requirement is stated as relative velocity and the deliverable is the combined velocity response itself.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "relative-velocity",
            "Select relativeVelocity for a response spectrum",
            "response = fea.SpectrumResponse.relativeVelocity();",
            "assert(strcmp(class(response), 'fea.SpectrumResponse'));\nassert(strcmp(response.value, 'relative_velocity'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which spectrum types give stress and displacement fields?",
            "Relative displacement, pseudo-velocity, and pseudo-acceleration spectra. RunMat converts their modal peaks to modal displacement (directly, divided by omega, or divided by omega squared) and recovers deformation fields; the other response quantities are combined but not converted.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.SpectrumResponse.relativeDisplacement",
        "fea.SpectrumResponse.relativeAcceleration",
        "fea.SpectrumResponse.absoluteAcceleration",
        "fea.SpectrumResponse.pseudoVelocity",
        "fea.SpectrumResponse.pseudoAcceleration",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumResponse.relativeVelocity",
        summary: "Read spectrum ordinates as peak relative velocity in meters per second.",
        description: "`fea.SpectrumResponse.relativeVelocity()` returns the closed `fea.SpectrumResponse` value `relative_velocity` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumResponse",
            "relativeVelocity",
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
        name: "fea.SpectrumResponse.relativeVelocity",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_RESPONSE,
        signature: "response = fea.SpectrumResponse.relativeVelocity()",
        inputs: [],
        output: ("response", "Typed fea.SpectrumResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod relative_acceleration {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each ordinate is the peak acceleration of a damped oscillator relative to its base, in m/s^2. It excludes the base acceleration itself.",
                "RunMat combines the modal peaks of this quantity, but it does not convert them to displacement: displacement-derived fields (deformation, strain, stress) are reported as unavailable for this response, and the result says so instead of guessing.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it only when a specification states relative acceleration; most shock and seismic spectra are absolute or pseudo-acceleration.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "relative-acceleration",
            "Select relativeAcceleration for a response spectrum",
            "response = fea.SpectrumResponse.relativeAcceleration();",
            "assert(strcmp(class(response), 'fea.SpectrumResponse'));\nassert(strcmp(response.value, 'relative_acceleration'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which spectrum types give stress and displacement fields?",
            "Relative displacement, pseudo-velocity, and pseudo-acceleration spectra. RunMat converts their modal peaks to modal displacement (directly, divided by omega, or divided by omega squared) and recovers deformation fields; the other response quantities are combined but not converted.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.SpectrumResponse.relativeDisplacement",
        "fea.SpectrumResponse.relativeVelocity",
        "fea.SpectrumResponse.absoluteAcceleration",
        "fea.SpectrumResponse.pseudoVelocity",
        "fea.SpectrumResponse.pseudoAcceleration",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumResponse.relativeAcceleration",
        summary: "Read spectrum ordinates as peak relative acceleration in meters per second squared.",
        description: "`fea.SpectrumResponse.relativeAcceleration()` returns the closed `fea.SpectrumResponse` value `relative_acceleration` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumResponse",
            "relativeAcceleration",
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
        name: "fea.SpectrumResponse.relativeAcceleration",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_RESPONSE,
        signature: "response = fea.SpectrumResponse.relativeAcceleration()",
        inputs: [],
        output: ("response", "Typed fea.SpectrumResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &relative_displacement::CATALOG_ENTRY,
    &relative_velocity::CATALOG_ENTRY,
    &relative_acceleration::CATALOG_ENTRY,
];
