//! `fea.StateUse` choices for bases, interfaces, and restarts.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod modal_basis {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumed modal-basis state supplies the modes for modal superposition, modal harmonic, response-spectrum, or random-response steps. Only modal-basis states are accepted, and modal-reduced steps require such an input.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it on every modal-reduced step, pointing at the modal extraction step.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "modal-basis",
            "Use modalBasis in a sequence",
            "value = fea.StateUse.modalBasis();\ninput = fea.StateInput.upstream(\"modal_basis\", fea.StateKind.modalBasis(), \"producer\", \"state\", ...\n    value, fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'modal_basis'));\nassert(strcmp(class(input), 'fea.StateInput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.StateTransition.new",
        "fea.StateTransitionMethod.reuse",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateUse.modalBasis",
        summary: "Consume a state as the modal basis of a modal-reduced step.",
        description: "`fea.StateUse.modalBasis()` returns the closed `fea.StateUse` value `modal_basis`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "modalBasis",
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
        name: "fea.StateUse.modalBasis",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.modalBasis()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod coupled_interface {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumed coupled-interface state supplies exchanged interface data to a partitioned coupled step. Only coupled-interface states are accepted.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for partitioned multiphysics sequences.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "coupled-interface",
            "Use coupledInterface in a sequence",
            "value = fea.StateUse.coupledInterface();\ninput = fea.StateInput.upstream(\"coupled_interface\", fea.StateKind.coupledInterface(), \"producer\", \"state\", ...\n    value, fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'coupled_interface'));\nassert(strcmp(class(input), 'fea.StateInput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.StateTransition.new",
        "fea.StateTransitionMethod.reuse",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateUse.coupledInterface",
        summary: "Consume a state at a coupled interface.",
        description: "`fea.StateUse.coupledInterface()` returns the closed `fea.StateUse` value `coupled_interface`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "coupledInterface",
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
        name: "fea.StateUse.coupledInterface",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.coupledInterface()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod restart {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumed state is the complete, persisted state a step continues from. `fea.RestartPolicy.continueFromCompletedState(stepId, stateId)` requires a matching restart input from the same producer and state.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a step continues another step's solution rather than starting fresh.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "restart",
            "Use restart in a sequence",
            "value = fea.StateUse.restart();\ninput = fea.StateInput.upstream(\"restart\", fea.StateKind.structuralPrimary(), \"producer\", \"state\", ...\n    value, fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'restart'));\nassert(strcmp(class(input), 'fea.StateInput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.StateTransition.new",
        "fea.StateTransitionMethod.reuse",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateUse.restart",
        summary: "Consume a state to continue a step from a completed state.",
        description: "`fea.StateUse.restart()` returns the closed `fea.StateUse` value `restart`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "restart",
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
        name: "fea.StateUse.restart",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.restart()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &modal_basis::CATALOG_ENTRY,
    &coupled_interface::CATALOG_ENTRY,
    &restart::CATALOG_ENTRY,
];
