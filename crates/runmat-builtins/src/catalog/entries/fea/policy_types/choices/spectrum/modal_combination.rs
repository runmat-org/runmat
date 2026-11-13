//! `fea.ModalCombination` choices: how modal peaks combine.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod srss {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each response component is `sqrt(sum_i R_i^2)` over the retained modes, assuming the modal peaks are statistically independent.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when natural frequencies are well separated.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "srss",
            "Select srss for a response spectrum",
            "combination = fea.ModalCombination.srss();",
            "assert(strcmp(class(combination), 'fea.ModalCombination'));\nassert(strcmp(combination.value, 'srss'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "When is SRSS not enough?",
            "When modes are closely spaced (frequencies within about 10 percent) their peaks are correlated and SRSS can be unconservative. Use completeQuadratic, or tenPercent where a code requires it.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.ModalCombination.completeQuadratic",
        "fea.ModalCombination.tenPercent",
        "fea.ModalCombination.absoluteSum",
        "fea.ModalTruncation.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ModalCombination.srss",
        summary: "Combine modal peaks by square root of the sum of squares.",
        description: "`fea.ModalCombination.srss()` returns the closed `fea.ModalCombination` value `srss` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "ModalCombination",
            "srss",
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
        name: "fea.ModalCombination.srss",
        result: crate::fea_contract::policy_types::classes::MODAL_COMBINATION,
        signature: "combination = fea.ModalCombination.srss()",
        inputs: [],
        output: ("combination", "Typed fea.ModalCombination value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod complete_quadratic {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each response component is `sqrt(sum_i sum_j rho_ij R_i R_j)` with the Der Kiureghian correlation coefficient computed from the two modal frequencies and the spectrum damping ratio (a quality factor Q becomes zeta = 1/(2Q)). Identical frequencies are fully correlated.",
                "Rigid-response separation (`fea.RigidResponse.gupta` or `.lindleyYow`) requires this rule.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it as the general-purpose rule, and always when modes are closely spaced.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "complete-quadratic",
            "Select completeQuadratic for a response spectrum",
            "combination = fea.ModalCombination.completeQuadratic();",
            "assert(strcmp(class(combination), 'fea.ModalCombination'));\nassert(strcmp(combination.value, 'complete_quadratic_combination'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "When is SRSS not enough?",
            "When modes are closely spaced (frequencies within about 10 percent) their peaks are correlated and SRSS can be unconservative. Use completeQuadratic, or tenPercent where a code requires it.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.ModalCombination.srss",
        "fea.ModalCombination.tenPercent",
        "fea.ModalCombination.absoluteSum",
        "fea.ModalTruncation.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ModalCombination.completeQuadratic",
        summary: "Combine modal peaks with the complete quadratic combination (CQC) rule.",
        description: "`fea.ModalCombination.completeQuadratic()` returns the closed `fea.ModalCombination` value `complete_quadratic_combination` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "ModalCombination",
            "completeQuadratic",
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
        name: "fea.ModalCombination.completeQuadratic",
        result: crate::fea_contract::policy_types::classes::MODAL_COMBINATION,
        signature: "combination = fea.ModalCombination.completeQuadratic()",
        inputs: [],
        output: ("combination", "Typed fea.ModalCombination value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod ten_percent {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each response component is `sqrt(sum_i sum_j rho_ij R_i R_j)` with `rho_ij = 1` when the higher of the two frequencies is within 10 percent of the lower one and `0` otherwise, so closely spaced modes add absolutely and others combine by SRSS.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a nuclear or regulatory code prescribes the ten-percent method.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "ten-percent",
            "Select tenPercent for a response spectrum",
            "combination = fea.ModalCombination.tenPercent();",
            "assert(strcmp(class(combination), 'fea.ModalCombination'));\nassert(strcmp(combination.value, 'ten_percent'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "When is SRSS not enough?",
            "When modes are closely spaced (frequencies within about 10 percent) their peaks are correlated and SRSS can be unconservative. Use completeQuadratic, or tenPercent where a code requires it.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.ModalCombination.srss",
        "fea.ModalCombination.completeQuadratic",
        "fea.ModalCombination.absoluteSum",
        "fea.ModalTruncation.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ModalCombination.tenPercent",
        summary: "Combine modal peaks with the ten-percent rule for closely spaced modes.",
        description: "`fea.ModalCombination.tenPercent()` returns the closed `fea.ModalCombination` value `ten_percent` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "ModalCombination",
            "tenPercent",
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
        name: "fea.ModalCombination.tenPercent",
        result: crate::fea_contract::policy_types::classes::MODAL_COMBINATION,
        signature: "combination = fea.ModalCombination.tenPercent()",
        inputs: [],
        output: ("combination", "Typed fea.ModalCombination value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod absolute_sum {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Each response component is `sum_i |R_i|`, assuming every modal peak occurs at the same instant with the same sign. It is an upper bound on the other rules.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for conservative bounding checks or when a specification requires absolute summation.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "absolute-sum",
            "Select absoluteSum for a response spectrum",
            "combination = fea.ModalCombination.absoluteSum();",
            "assert(strcmp(class(combination), 'fea.ModalCombination'));\nassert(strcmp(combination.value, 'absolute_sum'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "When is SRSS not enough?",
            "When modes are closely spaced (frequencies within about 10 percent) their peaks are correlated and SRSS can be unconservative. Use completeQuadratic, or tenPercent where a code requires it.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.ModalCombination.srss",
        "fea.ModalCombination.completeQuadratic",
        "fea.ModalCombination.tenPercent",
        "fea.ModalTruncation.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ModalCombination.absoluteSum",
        summary: "Combine modal peaks by absolute summation.",
        description: "`fea.ModalCombination.absoluteSum()` returns the closed `fea.ModalCombination` value `absolute_sum` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "ModalCombination",
            "absoluteSum",
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
        name: "fea.ModalCombination.absoluteSum",
        result: crate::fea_contract::policy_types::classes::MODAL_COMBINATION,
        signature: "combination = fea.ModalCombination.absoluteSum()",
        inputs: [],
        output: ("combination", "Typed fea.ModalCombination value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &srss::CATALOG_ENTRY,
    &complete_quadratic::CATALOG_ENTRY,
    &ten_percent::CATALOG_ENTRY,
    &absolute_sum::CATALOG_ENTRY,
];
