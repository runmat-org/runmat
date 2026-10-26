//! `fea.SpectrumResponse` choices for absolute and pseudo response quantities.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod absolute_acceleration {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each ordinate is the peak total acceleration of a damped oscillator, base motion included, in m/s^2. A shock response spectrum (SRS) derived from a measured acceleration history is of this type.",
                "Absolute acceleration admits missing-mass correction and rigid-response separation. RunMat combines the modal peaks of this quantity, but it does not convert them to displacement: displacement-derived fields (deformation, strain, stress) are reported as unavailable for this response, and the result says so instead of guessing.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for shock response spectra and equipment qualification spectra stated as total acceleration; choose pseudoAcceleration when you need stress fields from the same data.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "absolute-acceleration",
            "Select absoluteAcceleration for a response spectrum",
            "response = fea.SpectrumResponse.absoluteAcceleration();",
            "assert(strcmp(class(response), 'fea.SpectrumResponse'));\nassert(strcmp(response.value, 'absolute_acceleration'));",
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
        "fea.SpectrumResponse.relativeAcceleration",
        "fea.SpectrumResponse.pseudoVelocity",
        "fea.SpectrumResponse.pseudoAcceleration",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumResponse.absoluteAcceleration",
        summary: "Read spectrum ordinates as peak absolute (total) acceleration in meters per second squared.",
        description: "`fea.SpectrumResponse.absoluteAcceleration()` returns the closed `fea.SpectrumResponse` value `absolute_acceleration` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumResponse",
            "absoluteAcceleration",
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
        name: "fea.SpectrumResponse.absoluteAcceleration",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_RESPONSE,
        signature: "response = fea.SpectrumResponse.absoluteAcceleration()",
        inputs: [],
        output: ("response", "Typed fea.SpectrumResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod pseudo_velocity {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Pseudo-velocity is omega times the relative displacement, `PSV = 2*pi*f*SD`, in m/s.",
                "RunMat divides each mode's peak by its angular frequency to obtain modal displacement, so deformation, strain, and stress fields are recovered.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for tripartite design spectra read on the velocity axis.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "pseudo-velocity",
            "Select pseudoVelocity for a response spectrum",
            "response = fea.SpectrumResponse.pseudoVelocity();",
            "assert(strcmp(class(response), 'fea.SpectrumResponse'));\nassert(strcmp(response.value, 'pseudo_velocity'));",
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
        "fea.SpectrumResponse.relativeAcceleration",
        "fea.SpectrumResponse.absoluteAcceleration",
        "fea.SpectrumResponse.pseudoAcceleration",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumResponse.pseudoVelocity",
        summary: "Read spectrum ordinates as pseudo-velocity in meters per second.",
        description: "`fea.SpectrumResponse.pseudoVelocity()` returns the closed `fea.SpectrumResponse` value `pseudo_velocity` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumResponse",
            "pseudoVelocity",
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
        name: "fea.SpectrumResponse.pseudoVelocity",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_RESPONSE,
        signature: "response = fea.SpectrumResponse.pseudoVelocity()",
        inputs: [],
        output: ("response", "Typed fea.SpectrumResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod pseudo_acceleration {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Pseudo-acceleration is omega squared times the relative displacement, `PSA = (2*pi*f)^2*SD`, in m/s^2.",
                "RunMat divides each mode's peak by its angular frequency squared to obtain modal displacement and recovers deformation, strain, and stress fields. It is the only response for which a missing-mass static correction also yields recovered displacement fields.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for seismic design spectra, which are conventionally pseudo-acceleration spectra, and whenever you need stress from an acceleration spectrum.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "pseudo-acceleration",
            "Select pseudoAcceleration for a response spectrum",
            "response = fea.SpectrumResponse.pseudoAcceleration();",
            "assert(strcmp(class(response), 'fea.SpectrumResponse'));\nassert(strcmp(response.value, 'pseudo_acceleration'));",
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
        "fea.SpectrumResponse.relativeAcceleration",
        "fea.SpectrumResponse.absoluteAcceleration",
        "fea.SpectrumResponse.pseudoVelocity",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SpectrumResponse.pseudoAcceleration",
        summary: "Read spectrum ordinates as pseudo-acceleration in meters per second squared.",
        description: "`fea.SpectrumResponse.pseudoAcceleration()` returns the closed `fea.SpectrumResponse` value `pseudo_acceleration` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "SpectrumResponse",
            "pseudoAcceleration",
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
        name: "fea.SpectrumResponse.pseudoAcceleration",
        result: crate::fea_contract::policy_types::classes::SPECTRUM_RESPONSE,
        signature: "response = fea.SpectrumResponse.pseudoAcceleration()",
        inputs: [],
        output: ("response", "Typed fea.SpectrumResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &absolute_acceleration::CATALOG_ENTRY,
    &pseudo_velocity::CATALOG_ENTRY,
    &pseudo_acceleration::CATALOG_ENTRY,
];
