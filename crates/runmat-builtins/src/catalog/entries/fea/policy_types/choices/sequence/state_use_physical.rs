//! `fea.StateUse` choices that change the consumer's physics.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod physical_initial_state {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumed state is the step's starting condition, for example the temperature field at the start of a transient. Any state kind is accepted.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for initial conditions and for continuing physics from an upstream step.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "physical-initial-state",
            "Use physicalInitialState in a sequence",
            "value = fea.StateUse.physicalInitialState();\ninput = fea.StateInput.initialConditions(\"physical_initial_state\", fea.StateKind.thermal(), value, ...\n    fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'physical_initial_state'));\nassert(strcmp(class(input), 'fea.StateInput'));",
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
        title: "fea.StateUse.physicalInitialState",
        summary: "Consume a state as the physical initial state of a step.",
        description: "`fea.StateUse.physicalInitialState()` returns the closed `fea.StateUse` value `physical_initial_state`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "physicalInitialState",
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
        name: "fea.StateUse.physicalInitialState",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.physicalInitialState()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod linearization_point {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer forms its operator around the consumed state, as in a perturbation analysis. Any state kind is accepted; the matching transition method is `linearize` with an explicit linearization point.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for perturbation steps that need the tangent of a preloaded state.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "linearization-point",
            "Use linearizationPoint in a sequence",
            "value = fea.StateUse.linearizationPoint();\ninput = fea.StateInput.upstream(\"linearization_point\", fea.StateKind.structuralPrimary(), \"producer\", \"state\", ...\n    value, fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'linearization_point'));\nassert(strcmp(class(input), 'fea.StateInput'));",
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
        title: "fea.StateUse.linearizationPoint",
        summary: "Consume a state as the point a step linearizes about.",
        description: "`fea.StateUse.linearizationPoint()` returns the closed `fea.StateUse` value `linearization_point`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "linearizationPoint",
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
        name: "fea.StateUse.linearizationPoint",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.linearizationPoint()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod prestress {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumed state adds a stress-dependent contribution to the consumer's operator. Only structural-primary, structural stress-strain, geometric-stiffness, and thermal states are accepted. Prestressed modal extraction, eigenvalue buckling, and prestressed harmonic response require a geometric-stiffness prestress input; a thermomechanical static step consumes a solved thermal state this way.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to connect a preload or thermal step to a prestressed analysis.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "prestress",
            "Use prestress in a sequence",
            "value = fea.StateUse.prestress();\ninput = fea.StateInput.upstream(\"prestress\", fea.StateKind.geometricStiffness(), \"producer\", \"state\", ...\n    value, fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'prestress'));\nassert(strcmp(class(input), 'fea.StateInput'));",
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
        title: "fea.StateUse.prestress",
        summary: "Consume a state as prestress.",
        description: "`fea.StateUse.prestress()` returns the closed `fea.StateUse` value `prestress`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "prestress",
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
        name: "fea.StateUse.prestress",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.prestress()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod initial_guess {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumed state seeds the consumer's solver but does not change its physics; the converged answer does not depend on it. The transition must use `initialGuessOnly`, and vice versa. Structural execution supports a structural-primary initial guess.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to speed up a related solve without coupling the physics.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "initial-guess",
            "Use initialGuess in a sequence",
            "value = fea.StateUse.initialGuess();\ninput = fea.StateInput.upstream(\"initial_guess\", fea.StateKind.structuralPrimary(), \"producer\", \"state\", ...\n    value, fea.InitialStateConflictPolicy.reject());",
            "assert(strcmp(class(value), 'fea.StateUse'));\nassert(strcmp(value.value, 'initial_guess'));\nassert(strcmp(class(input), 'fea.StateInput'));",
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
        title: "fea.StateUse.initialGuess",
        summary: "Consume a state only as an initial guess.",
        description: "`fea.StateUse.initialGuess()` returns the closed `fea.StateUse` value `initial_guess`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateUse",
            "initialGuess",
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
        name: "fea.StateUse.initialGuess",
        result: crate::fea_contract::study_types::classes::FEA_STATE_USE_CLASS,
        signature: "value = fea.StateUse.initialGuess()",
        inputs: [],
        output: ("value", "Typed fea.StateUse value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &physical_initial_state::CATALOG_ENTRY,
    &linearization_point::CATALOG_ENTRY,
    &prestress::CATALOG_ENTRY,
    &initial_guess::CATALOG_ENTRY,
];
