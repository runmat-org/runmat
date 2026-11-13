//! `fea.StateTransitionMethod` choices that derive from a state.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod linearize {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer forms its operator around the transferred state. Validation requires `fea.ReferenceStatePolicy.explicitLinearizationPoint()`. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare a perturbation step about a preloaded state.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "linearize",
            "Use linearize in a sequence",
            "value = fea.StateTransitionMethod.linearize();\ntransition = fea.StateTransition.new(\"linearize_transfer\", \"producer\", \"consumer\", \"state\", ...\n    value, fea.TopologyTransitionPolicy.requireIdentical(), fea.ReferenceStatePolicy.explicitLinearizationPoint());",
            "assert(strcmp(class(value), 'fea.StateTransitionMethod'));\nassert(strcmp(value.value, 'linearize'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransitionMethod.linearize",
        summary: "Linearize the consumer about the producer state.",
        description: "`fea.StateTransitionMethod.linearize()` returns the closed `fea.StateTransitionMethod` value `linearize`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateTransitionMethod",
            "linearize",
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
        name: "fea.StateTransitionMethod.linearize",
        result: crate::fea_contract::study_types::classes::FEA_STATE_TRANSITION_METHOD_CLASS,
        signature: "value = fea.StateTransitionMethod.linearize()",
        inputs: [],
        output: ("value", "Typed fea.StateTransitionMethod value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod derive_operator_contribution {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer derives an operator term from the transferred state instead of using it as a field. Validation requires an explicit linearization point. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare stress-stiffening transfers into prestressed steps.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "derive-operator-contribution",
            "Use deriveOperatorContribution in a sequence",
            "value = fea.StateTransitionMethod.deriveOperatorContribution();\ntransition = fea.StateTransition.new(\"derive_operator_contribution_transfer\", \"producer\", \"consumer\", \"state\", ...\n    value, fea.TopologyTransitionPolicy.requireIdentical(), fea.ReferenceStatePolicy.explicitLinearizationPoint());",
            "assert(strcmp(class(value), 'fea.StateTransitionMethod'));\nassert(strcmp(value.value, 'derive_operator_contribution'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransitionMethod.deriveOperatorContribution",
        summary: "Derive an operator contribution, such as geometric stiffness, from the state.",
        description: "`fea.StateTransitionMethod.deriveOperatorContribution()` returns the closed `fea.StateTransitionMethod` value `derive_operator_contribution`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateTransitionMethod",
            "deriveOperatorContribution",
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
        name: "fea.StateTransitionMethod.deriveOperatorContribution",
        result: crate::fea_contract::study_types::classes::FEA_STATE_TRANSITION_METHOD_CLASS,
        signature: "value = fea.StateTransitionMethod.deriveOperatorContribution()",
        inputs: [],
        output: ("value", "Typed fea.StateTransitionMethod value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod initial_guess_only {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer uses the state to start its solver and does not depend on it physically. It must pair with an `initialGuess` state use. RunMat executes it on identical topology with the reference state preserved.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to warm-start a related structural solve.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "initial-guess-only",
            "Use initialGuessOnly in a sequence",
            "value = fea.StateTransitionMethod.initialGuessOnly();\ntransition = fea.StateTransition.new(\"initial_guess_only_transfer\", \"producer\", \"consumer\", \"state\", ...\n    value, fea.TopologyTransitionPolicy.requireIdentical(), fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.StateTransitionMethod'));\nassert(strcmp(value.value, 'initial_guess_only'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransitionMethod.initialGuessOnly",
        summary: "Transfer the state only as a solver initial guess.",
        description: "`fea.StateTransitionMethod.initialGuessOnly()` returns the closed `fea.StateTransitionMethod` value `initial_guess_only`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateTransitionMethod",
            "initialGuessOnly",
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
        name: "fea.StateTransitionMethod.initialGuessOnly",
        result: crate::fea_contract::study_types::classes::FEA_STATE_TRANSITION_METHOD_CLASS,
        signature: "value = fea.StateTransitionMethod.initialGuessOnly()",
        inputs: [],
        output: ("value", "Typed fea.StateTransitionMethod value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &linearize::CATALOG_ENTRY,
    &derive_operator_contribution::CATALOG_ENTRY,
    &initial_guess_only::CATALOG_ENTRY,
];
