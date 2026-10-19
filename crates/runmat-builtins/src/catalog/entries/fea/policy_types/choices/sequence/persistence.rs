//! `fea.StatePersistence` choices.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod canonical {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A canonical state is part of the study's durable result and remains available after the sequence finishes. Branches require complete canonical states.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for states you want to inspect, reuse, or branch from later.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "canonical",
            "Use canonical in a sequence",
            "value = fea.StatePersistence.canonical();\noutput = fea.StateOutput.new(\"static_equilibrium\", fea.StateKind.structuralPrimary(), fea.StateCompleteness.complete(), value);",
            "assert(strcmp(class(value), 'fea.StatePersistence'));\nassert(strcmp(value.value, 'canonical'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StatePersistence.canonical",
        summary: "Retain a published state canonically.",
        description: "`fea.StatePersistence.canonical()` returns the closed `fea.StatePersistence` value `canonical`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StatePersistence",
            "canonical",
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
        name: "fea.StatePersistence.canonical",
        result: crate::fea_contract::study_types::classes::FEA_STATE_PERSISTENCE_CLASS,
        signature: "value = fea.StatePersistence.canonical()",
        inputs: [],
        output: ("value", "Typed fea.StatePersistence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod until_dependents_complete {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The state is kept while dependent steps may still read it and can be released afterward. It can still cross procedure boundaries but cannot be published by a branch. Steady fluid execution requires this persistence for its fluid output.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for large intermediate states, such as a flow field feeding a thermal step, that are not part of the deliverable.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "until-dependents-complete",
            "Use untilDependentsComplete in a sequence",
            "value = fea.StatePersistence.untilDependentsComplete();\noutput = fea.StateOutput.new(\"flow\", fea.StateKind.fluid(), fea.StateCompleteness.complete(), value);",
            "assert(strcmp(class(value), 'fea.StatePersistence'));\nassert(strcmp(value.value, 'until_dependents_complete'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StatePersistence.untilDependentsComplete",
        summary: "Retain a published state only until its consumers finish.",
        description: "`fea.StatePersistence.untilDependentsComplete()` returns the closed `fea.StatePersistence` value `until_dependents_complete`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StatePersistence",
            "untilDependentsComplete",
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
        name: "fea.StatePersistence.untilDependentsComplete",
        result: crate::fea_contract::study_types::classes::FEA_STATE_PERSISTENCE_CLASS,
        signature: "value = fea.StatePersistence.untilDependentsComplete()",
        inputs: [],
        output: ("value", "Typed fea.StatePersistence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod checkpoint_only {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The state is stored only to resume the producing step. Checkpoint completeness requires it; a complete state cannot use it; it cannot be transferred, continued from, or published by a branch.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it together with `fea.StateCompleteness.checkpoint()` for restart data.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "checkpoint-only",
            "Use checkpointOnly in a sequence",
            "value = fea.StatePersistence.checkpointOnly();\noutput = fea.StateOutput.new(\"integrator_checkpoint\", fea.StateKind.integrator(), fea.StateCompleteness.checkpoint(), value);",
            "assert(strcmp(class(value), 'fea.StatePersistence'));\nassert(strcmp(value.value, 'checkpoint_only'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StatePersistence.checkpointOnly",
        summary: "Retain a published state only as a checkpoint.",
        description: "`fea.StatePersistence.checkpointOnly()` returns the closed `fea.StatePersistence` value `checkpoint_only`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StatePersistence",
            "checkpointOnly",
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
        name: "fea.StatePersistence.checkpointOnly",
        result: crate::fea_contract::study_types::classes::FEA_STATE_PERSISTENCE_CLASS,
        signature: "value = fea.StatePersistence.checkpointOnly()",
        inputs: [],
        output: ("value", "Typed fea.StatePersistence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &canonical::CATALOG_ENTRY,
    &until_dependents_complete::CATALOG_ENTRY,
    &checkpoint_only::CATALOG_ENTRY,
];
