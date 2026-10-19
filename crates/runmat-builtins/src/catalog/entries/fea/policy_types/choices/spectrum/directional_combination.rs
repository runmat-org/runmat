//! `fea.DirectionalCombination` choices: how excitation directions combine.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod srss {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The modally combined responses of the spectra applied in different directions are combined component by component as `sqrt(sum_d R_d^2)`.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for simultaneous, statistically independent excitation in two or three directions, as most seismic codes allow.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "srss",
            "Select srss for a response spectrum",
            "combination = fea.DirectionalCombination.srss();",
            "assert(strcmp(class(combination), 'fea.DirectionalCombination'));\nassert(strcmp(combination.value, 'srss'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.DirectionalCombination.absoluteSum",
        "fea.DirectionalCombination.percentage",
        "fea.SpectrumDirection.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.DirectionalCombination.srss",
        summary: "Combine responses from different excitation directions by SRSS.",
        description: "`fea.DirectionalCombination.srss()` returns the closed `fea.DirectionalCombination` value `srss` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "DirectionalCombination",
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
        name: "fea.DirectionalCombination.srss",
        result: crate::fea_contract::policy_types::classes::DIRECTIONAL_COMBINATION,
        signature: "combination = fea.DirectionalCombination.srss()",
        inputs: [],
        output: ("combination", "Typed fea.DirectionalCombination value."),
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
                "The directional responses are combined component by component as `sum_d |R_d|`, assuming the directional peaks coincide.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it as a conservative bound; use `fea.DirectionalCombination.percentage` for 100-30 or 100-40 percentage rules.",
            ],
        ),
        USE_SPECTRUM,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "absolute-sum",
            "Select absoluteSum for a response spectrum",
            "combination = fea.DirectionalCombination.absoluteSum();",
            "assert(strcmp(class(combination), 'fea.DirectionalCombination'));\nassert(strcmp(combination.value, 'absolute_sum'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.DirectionalCombination.srss",
        "fea.DirectionalCombination.percentage",
        "fea.SpectrumDirection.new",
        "fea.ResponseSpectrum.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.DirectionalCombination.absoluteSum",
        summary: "Combine responses from different excitation directions by absolute summation.",
        description: "`fea.DirectionalCombination.absoluteSum()` returns the closed `fea.DirectionalCombination` value `absolute_sum` for a response spectrum. It takes no arguments.",
        keywords: &[
            "fea",
            "response spectrum",
            "DirectionalCombination",
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
        name: "fea.DirectionalCombination.absoluteSum",
        result: crate::fea_contract::policy_types::classes::DIRECTIONAL_COMBINATION,
        signature: "combination = fea.DirectionalCombination.absoluteSum()",
        inputs: [],
        output: ("combination", "Typed fea.DirectionalCombination value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&srss::CATALOG_ENTRY, &absolute_sum::CATALOG_ENTRY];
