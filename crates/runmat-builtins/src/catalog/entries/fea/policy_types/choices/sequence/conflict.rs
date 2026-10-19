//! `fea.InitialStateConflictPolicy` choices.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod reject {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "If the upstream state and the step's authored initial conditions define the same quantity, validation fails instead of choosing one. It is the only policy allowed for inputs sourced from initial conditions.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it by default; it keeps every initial value traceable to exactly one definition.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "reject",
            "Use reject in a sequence",
            "value = fea.InitialStateConflictPolicy.reject();\ninput = fea.StateInput.upstream(\"reject\", fea.StateKind.thermal(), \"preheat\", \"temperature\", ...\n    fea.StateUse.physicalInitialState(), value);",
            "assert(strcmp(class(value), 'fea.InitialStateConflictPolicy'));\nassert(strcmp(value.value, 'reject'));\nassert(strcmp(class(input), 'fea.StateInput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Can initial-condition inputs override anything?",
            "No. An input whose source is the model's initial conditions must use `reject`; only upstream inputs may declare replacement or augmentation.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.InitialStateConflictPolicy.replaceInitialCondition",
        "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
        "fea.StateInput.upstream",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.InitialStateConflictPolicy.reject",
        summary: "Reject an upstream state that conflicts with authored initial conditions.",
        description: "`fea.InitialStateConflictPolicy.reject()` returns the closed `fea.InitialStateConflictPolicy` value `reject`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "InitialStateConflictPolicy",
            "reject",
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
        name: "fea.InitialStateConflictPolicy.reject",
        result: crate::fea_contract::study_types::classes::FEA_INITIAL_STATE_CONFLICT_POLICY_CLASS,
        signature: "value = fea.InitialStateConflictPolicy.reject()",
        inputs: [],
        output: ("value", "Typed fea.InitialStateConflictPolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod replace_initial_condition {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Where the upstream state and an authored initial condition overlap, the upstream state is declared to take precedence. It is allowed only on upstream inputs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a template model carries default initial conditions that a preceding step should supersede.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "replace-initial-condition",
            "Use replaceInitialCondition in a sequence",
            "value = fea.InitialStateConflictPolicy.replaceInitialCondition();\ninput = fea.StateInput.upstream(\"replace_initial_condition\", fea.StateKind.thermal(), \"preheat\", \"temperature\", ...\n    fea.StateUse.physicalInitialState(), value);",
            "assert(strcmp(class(value), 'fea.InitialStateConflictPolicy'));\nassert(strcmp(value.value, 'replace_initial_condition'));\nassert(strcmp(class(input), 'fea.StateInput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Can initial-condition inputs override anything?",
            "No. An input whose source is the model's initial conditions must use `reject`; only upstream inputs may declare replacement or augmentation.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.InitialStateConflictPolicy.reject",
        "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
        "fea.StateInput.upstream",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.InitialStateConflictPolicy.replaceInitialCondition",
        summary: "Let the upstream state replace a conflicting initial condition.",
        description: "`fea.InitialStateConflictPolicy.replaceInitialCondition()` returns the closed `fea.InitialStateConflictPolicy` value `replace_initial_condition`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "InitialStateConflictPolicy",
            "replaceInitialCondition",
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
        name: "fea.InitialStateConflictPolicy.replaceInitialCondition",
        result: crate::fea_contract::study_types::classes::FEA_INITIAL_STATE_CONFLICT_POLICY_CLASS,
        signature: "value = fea.InitialStateConflictPolicy.replaceInitialCondition()",
        inputs: [],
        output: ("value", "Typed fea.InitialStateConflictPolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod augment_disjoint_quantities {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The upstream state and authored initial conditions may both contribute as long as they define different quantities; overlapping definitions are still a conflict. It is allowed only on upstream inputs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when, for example, temperature comes from an upstream thermal step while velocity comes from authored initial conditions.",
            ],
        ),
        USE_STATE_INPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "augment-disjoint-quantities",
            "Use augmentDisjointQuantities in a sequence",
            "value = fea.InitialStateConflictPolicy.augmentDisjointQuantities();\ninput = fea.StateInput.upstream(\"augment_disjoint_quantities\", fea.StateKind.thermal(), \"preheat\", \"temperature\", ...\n    fea.StateUse.physicalInitialState(), value);",
            "assert(strcmp(class(value), 'fea.InitialStateConflictPolicy'));\nassert(strcmp(value.value, 'augment_disjoint_quantities'));\nassert(strcmp(class(input), 'fea.StateInput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Can initial-condition inputs override anything?",
            "No. An input whose source is the model's initial conditions must use `reject`; only upstream inputs may declare replacement or augmentation.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.InitialStateConflictPolicy.reject",
        "fea.InitialStateConflictPolicy.replaceInitialCondition",
        "fea.StateInput.upstream",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
        summary: "Combine upstream state with initial conditions only for disjoint quantities.",
        description: "`fea.InitialStateConflictPolicy.augmentDisjointQuantities()` returns the closed `fea.InitialStateConflictPolicy` value `augment_disjoint_quantities`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "InitialStateConflictPolicy",
            "augmentDisjointQuantities",
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
        name: "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
        result: crate::fea_contract::study_types::classes::FEA_INITIAL_STATE_CONFLICT_POLICY_CLASS,
        signature: "value = fea.InitialStateConflictPolicy.augmentDisjointQuantities()",
        inputs: [],
        output: ("value", "Typed fea.InitialStateConflictPolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &reject::CATALOG_ENTRY,
    &replace_initial_condition::CATALOG_ENTRY,
    &augment_disjoint_quantities::CATALOG_ENTRY,
];
