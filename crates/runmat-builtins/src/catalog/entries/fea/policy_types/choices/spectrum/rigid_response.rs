//! `fea.RigidResponse.disabled`: no rigid/periodic separation.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod disabled {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "All modal responses are combined by the selected modal combination rule; no part of any mode is treated as rigid (in phase with the input). The rigid coefficients and rigid response recorded with the result are zero.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it unless a code requires rigid-response treatment of high-frequency modes; `fea.RigidResponse.gupta` and `.lindleyYow` provide that for acceleration spectra with CQC.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "disabled",
            "Select disabled for a response spectrum",
            "policy = fea.RigidResponse.disabled();",
            "assert(strcmp(class(policy), 'fea.RigidResponse'));\nassert(strcmp(policy.value, 'disabled'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.RigidResponse.gupta",
        "fea.RigidResponse.lindleyYow",
        "fea.ModalCombination.completeQuadratic",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.RigidResponse.disabled",
        summary: "Do not separate rigid and periodic modal response.",
        description: "`fea.RigidResponse.disabled()` returns the closed `fea.RigidResponse` value `disabled` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "RigidResponse",
            "disabled",
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
        name: "fea.RigidResponse.disabled",
        result: crate::fea_contract::policy_types::classes::RIGID_RESPONSE,
        signature: "policy = fea.RigidResponse.disabled()",
        inputs: [],
        output: ("policy", "Typed fea.RigidResponse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&disabled::CATALOG_ENTRY];
