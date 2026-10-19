//! `fea.StateKind` choices for history and thermal states.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod material_history {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Material-history state carries path-dependent internal variables (plastic strain, hardening, damage) at integration points.",
                "Structural and thermomechanical steps can publish it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a following step must continue a path-dependent material response.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "material-history",
            "Use materialHistory in a sequence",
            "value = fea.StateKind.materialHistory();\noutput = fea.StateOutput.new(\"material_history\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'material_history'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.materialHistory",
        summary: "Declare material-history state such as plastic strain.",
        description: "`fea.StateKind.materialHistory()` returns the closed `fea.StateKind` value `material_history`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "materialHistory",
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
        name: "fea.StateKind.materialHistory",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.materialHistory()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod contact_history {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Contact-history state carries contact status, gaps, and accumulated slip between steps.",
                "Structural and thermomechanical steps can publish it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when frictional contact must continue from a previous step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "contact-history",
            "Use contactHistory in a sequence",
            "value = fea.StateKind.contactHistory();\noutput = fea.StateOutput.new(\"contact_history\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'contact_history'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.contactHistory",
        summary: "Declare contact-history state.",
        description: "`fea.StateKind.contactHistory()` returns the closed `fea.StateKind` value `contact_history`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "contactHistory",
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
        name: "fea.StateKind.contactHistory",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.contactHistory()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod thermal {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Thermal state is a solved temperature field.",
                "Thermal, thermomechanical, electrothermal, and conjugate heat transfer steps can publish it; structural steps can consume it, and a thermomechanical static step uses it as prestress.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to pass a solved temperature field to a thermal-stress step or as the initial state of a transient thermal step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "thermal",
            "Use thermal in a sequence",
            "value = fea.StateKind.thermal();\noutput = fea.StateOutput.new(\"thermal\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'thermal'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.thermal",
        summary: "Declare a thermal state (temperature field).",
        description: "`fea.StateKind.thermal()` returns the closed `fea.StateKind` value `thermal`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "thermal",
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
        name: "fea.StateKind.thermal",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.thermal()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &material_history::CATALOG_ENTRY,
    &contact_history::CATALOG_ENTRY,
    &thermal::CATALOG_ENTRY,
];
