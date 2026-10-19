//! `fea.StateTransitionMethod` choices that keep or map a state.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod reuse {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer receives the producer's state byte for byte. Validation requires identical topology. This is one of the two transfers RunMat executes today, with the reference state preserved.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it whenever producer and consumer share the same mesh, for example modal extraction followed by a modal-reduced step.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "reuse",
            "Use reuse in a sequence",
            "value = fea.StateTransitionMethod.reuse();\ntransition = fea.StateTransition.new(\"reuse_transfer\", \"producer\", \"consumer\", \"state\", ...\n    value, fea.TopologyTransitionPolicy.requireIdentical(), fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.StateTransitionMethod'));\nassert(strcmp(value.value, 'reuse'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransitionMethod.reuse",
        summary: "Reuse the producer state unchanged.",
        description: "`fea.StateTransitionMethod.reuse()` returns the closed `fea.StateTransitionMethod` value `reuse`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateTransitionMethod",
            "reuse",
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
        name: "fea.StateTransitionMethod.reuse",
        result: crate::fea_contract::study_types::classes::FEA_STATE_TRANSITION_METHOD_CLASS,
        signature: "value = fea.StateTransitionMethod.reuse()",
        inputs: [],
        output: ("value", "Typed fea.StateTransitionMethod value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod conservative_projection {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The state is projected so integral quantities (mass, energy, load) are preserved on the consumer's discretization. It is required, with interpolation, for adapted-mesh lineage. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare a conservation-preserving transfer onto a different mesh.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "conservative-projection",
            "Use conservativeProjection in a sequence",
            "value = fea.StateTransitionMethod.conservativeProjection();\ntransition = fea.StateTransition.new(\"conservative_projection_transfer\", \"producer\", \"consumer\", \"state\", ...\n    value, fea.TopologyTransitionPolicy.adaptedMeshLineage(), fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.StateTransitionMethod'));\nassert(strcmp(value.value, 'conservative_projection'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransitionMethod.conservativeProjection",
        summary: "Project the state onto the consumer conservatively.",
        description: "`fea.StateTransitionMethod.conservativeProjection()` returns the closed `fea.StateTransitionMethod` value `conservative_projection`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateTransitionMethod",
            "conservativeProjection",
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
        name: "fea.StateTransitionMethod.conservativeProjection",
        result: crate::fea_contract::study_types::classes::FEA_STATE_TRANSITION_METHOD_CLASS,
        signature: "value = fea.StateTransitionMethod.conservativeProjection()",
        inputs: [],
        output: ("value", "Typed fea.StateTransitionMethod value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod interpolation {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The state is interpolated pointwise onto the consumer's discretization, with transfer evidence. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare a pointwise transfer onto a projected or adapted mesh.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "interpolation",
            "Use interpolation in a sequence",
            "value = fea.StateTransitionMethod.interpolation();\ntransition = fea.StateTransition.new(\"interpolation_transfer\", \"producer\", \"consumer\", \"state\", ...\n    value, fea.TopologyTransitionPolicy.projectWithEvidence(), fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.StateTransitionMethod'));\nassert(strcmp(value.value, 'interpolation'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransitionMethod.interpolation",
        summary: "Interpolate the state onto the consumer with evidence.",
        description: "`fea.StateTransitionMethod.interpolation()` returns the closed `fea.StateTransitionMethod` value `interpolation`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateTransitionMethod",
            "interpolation",
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
        name: "fea.StateTransitionMethod.interpolation",
        result: crate::fea_contract::study_types::classes::FEA_STATE_TRANSITION_METHOD_CLASS,
        signature: "value = fea.StateTransitionMethod.interpolation()",
        inputs: [],
        output: ("value", "Typed fea.StateTransitionMethod value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &reuse::CATALOG_ENTRY,
    &conservative_projection::CATALOG_ENTRY,
    &interpolation::CATALOG_ENTRY,
];
