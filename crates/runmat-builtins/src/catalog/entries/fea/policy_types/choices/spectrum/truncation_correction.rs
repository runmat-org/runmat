//! `fea.ModalTruncationCorrection` choices without parameters.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod none {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The response comes from the retained eigenmodes only. The result still records the achieved cumulative effective-mass ratio against the minimum ratio required by `fea.ModalTruncation.new(minimumMassRatio, correction)`.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when the retained modes capture the required effective mass in every excitation direction.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "none",
            "Select none for a response spectrum",
            "correction = fea.ModalTruncationCorrection.none();\ntruncation = fea.ModalTruncation.new(0.9, correction);",
            "assert(strcmp(class(correction), 'fea.ModalTruncationCorrection'));\nassert(strcmp(correction.value, 'none'));\nassert(strcmp(class(truncation), 'fea.ModalTruncation'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.ModalTruncationCorrection.residualMode",
        "fea.ModalTruncationCorrection.missingMass",
        "fea.ModalTruncation.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ModalTruncationCorrection.none",
        summary: "Apply no correction for modes left out of the modal basis.",
        description: "`fea.ModalTruncationCorrection.none()` returns the closed `fea.ModalTruncationCorrection` value `none` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "ModalTruncationCorrection",
            "none",
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
        name: "fea.ModalTruncationCorrection.none",
        result: crate::fea_contract::policy_types::classes::MODAL_TRUNCATION_CORRECTION,
        signature: "correction = fea.ModalTruncationCorrection.none()",
        inputs: [],
        output: ("correction", "Typed fea.ModalTruncationCorrection value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod residual_mode {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Planning adds one static residual vector per excitation direction and the solve combines it with the eigenmodes as an extra pseudo-mode. The truncation evidence records one residual vector.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when high-frequency modes carry significant mass that the extracted modes do not capture, without needing a zero-period acceleration.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "residual-mode",
            "Select residualMode for a response spectrum",
            "correction = fea.ModalTruncationCorrection.residualMode();\ntruncation = fea.ModalTruncation.new(0.9, correction);",
            "assert(strcmp(class(correction), 'fea.ModalTruncationCorrection'));\nassert(strcmp(correction.value, 'residual_mode'));\nassert(strcmp(class(truncation), 'fea.ModalTruncation'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.ModalTruncationCorrection.none",
        "fea.ModalTruncationCorrection.missingMass",
        "fea.ModalTruncation.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ModalTruncationCorrection.residualMode",
        summary: "Add a load-dependent residual mode for each spectrum direction.",
        description: "`fea.ModalTruncationCorrection.residualMode()` returns the closed `fea.ModalTruncationCorrection` value `residual_mode` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "ModalTruncationCorrection",
            "residualMode",
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
        name: "fea.ModalTruncationCorrection.residualMode",
        result: crate::fea_contract::policy_types::classes::MODAL_TRUNCATION_CORRECTION,
        signature: "correction = fea.ModalTruncationCorrection.residualMode()",
        inputs: [],
        output: ("correction", "Typed fea.ModalTruncationCorrection value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&none::CATALOG_ENTRY, &residual_mode::CATALOG_ENTRY];
