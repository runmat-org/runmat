//! `fea.StateCompleteness` choices.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod complete {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A complete state contains everything needed to use it as a physical state. Only complete states can cross a procedure boundary through a transition, serve a continuation restart, or be published by a branch.",
                "A complete state cannot be `checkpointOnly`.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for every state a later step consumes.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "complete",
            "Use complete in a sequence",
            "value = fea.StateCompleteness.complete();\noutput = fea.StateOutput.new(\"static_equilibrium\", fea.StateKind.structuralPrimary(), value, fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateCompleteness'));\nassert(strcmp(value.value, 'complete'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateCompleteness.complete",
        summary: "Declare that a published state is complete.",
        description: "`fea.StateCompleteness.complete()` returns the closed `fea.StateCompleteness` value `complete`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateCompleteness",
            "complete",
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
        name: "fea.StateCompleteness.complete",
        result: crate::fea_contract::study_types::classes::FEA_STATE_COMPLETENESS_CLASS,
        signature: "value = fea.StateCompleteness.complete()",
        inputs: [],
        output: ("value", "Typed fea.StateCompleteness value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod checkpoint {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A checkpoint state lets the same step resume, for example a transient integrator history, but is not a physical state for other steps. It must use `checkpointOnly` persistence and cannot be transferred.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for integrator or solver history that only a restart of the same step needs.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "checkpoint",
            "Use checkpoint in a sequence",
            "value = fea.StateCompleteness.checkpoint();\noutput = fea.StateOutput.new(\"integrator_checkpoint\", fea.StateKind.integrator(), value, fea.StatePersistence.checkpointOnly());",
            "assert(strcmp(class(value), 'fea.StateCompleteness'));\nassert(strcmp(value.value, 'checkpoint'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateCompleteness.checkpoint",
        summary: "Declare that a published state is a resumable checkpoint.",
        description: "`fea.StateCompleteness.checkpoint()` returns the closed `fea.StateCompleteness` value `checkpoint`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateCompleteness",
            "checkpoint",
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
        name: "fea.StateCompleteness.checkpoint",
        result: crate::fea_contract::study_types::classes::FEA_STATE_COMPLETENESS_CLASS,
        signature: "value = fea.StateCompleteness.checkpoint()",
        inputs: [],
        output: ("value", "Typed fea.StateCompleteness value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod partial {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A partial state covers only part of the model or quantities. It is retained for inspection but cannot cross a procedure boundary, serve a continuation, or be published by a branch.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for diagnostic subsets of a state that no step consumes.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "partial",
            "Use partial in a sequence",
            "value = fea.StateCompleteness.partial();\noutput = fea.StateOutput.new(\"surface_temperature\", fea.StateKind.thermal(), value, fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateCompleteness'));\nassert(strcmp(value.value, 'partial'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateCompleteness.partial",
        summary: "Declare that a published state is partial.",
        description: "`fea.StateCompleteness.partial()` returns the closed `fea.StateCompleteness` value `partial`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateCompleteness",
            "partial",
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
        name: "fea.StateCompleteness.partial",
        result: crate::fea_contract::study_types::classes::FEA_STATE_COMPLETENESS_CLASS,
        signature: "value = fea.StateCompleteness.partial()",
        inputs: [],
        output: ("value", "Typed fea.StateCompleteness value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &complete::CATALOG_ENTRY,
    &checkpoint::CATALOG_ENTRY,
    &partial::CATALOG_ENTRY,
];
