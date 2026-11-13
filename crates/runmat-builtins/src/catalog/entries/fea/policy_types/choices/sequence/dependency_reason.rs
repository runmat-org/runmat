//! `fea.SequenceDependencyReason` choices.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod state_transfer {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The successor consumes state published by the predecessor. Every `fea.StateTransition` requires a direct dependency with this reason between its producer and consumer.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for every producer-consumer pair that has a state transition.",
            ],
        ),
        USE_DEPENDENCY,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "state-transfer",
            "Use stateTransfer in a sequence",
            "value = fea.SequenceDependencyReason.stateTransfer();\ndependency = fea.SequenceDependency.new(\"preload\", \"modes\", value);",
            "assert(strcmp(class(value), 'fea.SequenceDependencyReason'));\nassert(strcmp(value.value, 'state_transfer'));\nassert(strcmp(class(dependency), 'fea.SequenceDependency'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SequenceDependency.new",
        "fea.AnalysisSequence.new",
        "fea.StateTransition.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SequenceDependencyReason.stateTransfer",
        summary: "Order two steps because state flows between them.",
        description: "`fea.SequenceDependencyReason.stateTransfer()` returns the closed `fea.SequenceDependencyReason` value `state_transfer`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "SequenceDependencyReason",
            "stateTransfer",
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
        name: "fea.SequenceDependencyReason.stateTransfer",
        result: crate::fea_contract::study_types::classes::FEA_SEQUENCE_DEPENDENCY_REASON_CLASS,
        signature: "value = fea.SequenceDependencyReason.stateTransfer()",
        inputs: [],
        output: ("value", "Typed fea.SequenceDependencyReason value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod authored_ordering {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The successor runs after the predecessor without exchanging state, for example to keep a reporting order or to share an external resource.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when order matters for reasons outside the physics; it does not admit a state transition.",
            ],
        ),
        USE_DEPENDENCY,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "authored-ordering",
            "Use authoredOrdering in a sequence",
            "value = fea.SequenceDependencyReason.authoredOrdering();\ndependency = fea.SequenceDependency.new(\"preload\", \"modes\", value);",
            "assert(strcmp(class(value), 'fea.SequenceDependencyReason'));\nassert(strcmp(value.value, 'authored_ordering'));\nassert(strcmp(class(dependency), 'fea.SequenceDependency'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SequenceDependency.new",
        "fea.AnalysisSequence.new",
        "fea.StateTransition.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SequenceDependencyReason.authoredOrdering",
        summary: "Order two steps because the author requires it.",
        description: "`fea.SequenceDependencyReason.authoredOrdering()` returns the closed `fea.SequenceDependencyReason` value `authored_ordering`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "SequenceDependencyReason",
            "authoredOrdering",
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
        name: "fea.SequenceDependencyReason.authoredOrdering",
        result: crate::fea_contract::study_types::classes::FEA_SEQUENCE_DEPENDENCY_REASON_CLASS,
        signature: "value = fea.SequenceDependencyReason.authoredOrdering()",
        inputs: [],
        output: ("value", "Typed fea.SequenceDependencyReason value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod shared_interface {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The two steps act on the same interface, so the successor runs after the predecessor has finished with it. It does not by itself admit a state transition.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for partitioned coupled sequences that exchange data through an interface.",
            ],
        ),
        USE_DEPENDENCY,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "shared-interface",
            "Use sharedInterface in a sequence",
            "value = fea.SequenceDependencyReason.sharedInterface();\ndependency = fea.SequenceDependency.new(\"preload\", \"modes\", value);",
            "assert(strcmp(class(value), 'fea.SequenceDependencyReason'));\nassert(strcmp(value.value, 'shared_interface'));\nassert(strcmp(class(dependency), 'fea.SequenceDependency'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SequenceDependency.new",
        "fea.AnalysisSequence.new",
        "fea.StateTransition.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SequenceDependencyReason.sharedInterface",
        summary: "Order two steps that share a coupled interface.",
        description: "`fea.SequenceDependencyReason.sharedInterface()` returns the closed `fea.SequenceDependencyReason` value `shared_interface`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "SequenceDependencyReason",
            "sharedInterface",
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
        name: "fea.SequenceDependencyReason.sharedInterface",
        result: crate::fea_contract::study_types::classes::FEA_SEQUENCE_DEPENDENCY_REASON_CLASS,
        signature: "value = fea.SequenceDependencyReason.sharedInterface()",
        inputs: [],
        output: ("value", "Typed fea.SequenceDependencyReason value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod adaptation {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The successor runs on a mesh adapted from the predecessor's result. A state carried across the adaptation still needs a `stateTransfer` dependency and a transition with projection or interpolation.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to record adaptive refinement lineage between steps.",
            ],
        ),
        USE_DEPENDENCY,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "adaptation",
            "Use adaptation in a sequence",
            "value = fea.SequenceDependencyReason.adaptation();\ndependency = fea.SequenceDependency.new(\"preload\", \"modes\", value);",
            "assert(strcmp(class(value), 'fea.SequenceDependencyReason'));\nassert(strcmp(value.value, 'adaptation'));\nassert(strcmp(class(dependency), 'fea.SequenceDependency'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.SequenceDependency.new",
        "fea.AnalysisSequence.new",
        "fea.StateTransition.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.SequenceDependencyReason.adaptation",
        summary: "Order two steps across a mesh adaptation.",
        description: "`fea.SequenceDependencyReason.adaptation()` returns the closed `fea.SequenceDependencyReason` value `adaptation`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "SequenceDependencyReason",
            "adaptation",
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
        name: "fea.SequenceDependencyReason.adaptation",
        result: crate::fea_contract::study_types::classes::FEA_SEQUENCE_DEPENDENCY_REASON_CLASS,
        signature: "value = fea.SequenceDependencyReason.adaptation()",
        inputs: [],
        output: ("value", "Typed fea.SequenceDependencyReason value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &state_transfer::CATALOG_ENTRY,
    &authored_ordering::CATALOG_ENTRY,
    &shared_interface::CATALOG_ENTRY,
    &adaptation::CATALOG_ENTRY,
];
