//! `fea.ReferenceStatePolicy` choices.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod preserve {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Quantities in the transferred state stay measured from the producer's reference configuration. It is the only reference policy RunMat executes today.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for reuse and initial-guess transfers.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "preserve",
            "Use preserve in a sequence",
            "value = fea.ReferenceStatePolicy.preserve();\ntransition = fea.StateTransition.new(\"preserve_transfer\", \"producer\", \"consumer\", \"state\", ...\n    fea.StateTransitionMethod.reuse(), fea.TopologyTransitionPolicy.requireIdentical(), value);",
            "assert(strcmp(class(value), 'fea.ReferenceStatePolicy'));\nassert(strcmp(value.value, 'preserve'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ReferenceStatePolicy.preserve",
        summary: "Keep the producer's reference state.",
        description: "`fea.ReferenceStatePolicy.preserve()` returns the closed `fea.ReferenceStatePolicy` value `preserve`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "ReferenceStatePolicy",
            "preserve",
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
        name: "fea.ReferenceStatePolicy.preserve",
        result: crate::fea_contract::study_types::classes::FEA_REFERENCE_STATE_POLICY_CLASS,
        signature: "value = fea.ReferenceStatePolicy.preserve()",
        inputs: [],
        output: ("value", "Typed fea.ReferenceStatePolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod receiving_step_origin {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Quantities are re-expressed relative to the consumer's own origin, for example resetting accumulated displacement to zero at the start of the consumer. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare that a consumer measures its response from its own start.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "receiving-step-origin",
            "Use receivingStepOrigin in a sequence",
            "value = fea.ReferenceStatePolicy.receivingStepOrigin();\ntransition = fea.StateTransition.new(\"receiving_step_origin_transfer\", \"producer\", \"consumer\", \"state\", ...\n    fea.StateTransitionMethod.interpolation(), fea.TopologyTransitionPolicy.projectWithEvidence(), value);",
            "assert(strcmp(class(value), 'fea.ReferenceStatePolicy'));\nassert(strcmp(value.value, 'receiving_step_origin'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ReferenceStatePolicy.receivingStepOrigin",
        summary: "Re-reference the state to the receiving step's origin.",
        description: "`fea.ReferenceStatePolicy.receivingStepOrigin()` returns the closed `fea.ReferenceStatePolicy` value `receiving_step_origin`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "ReferenceStatePolicy",
            "receivingStepOrigin",
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
        name: "fea.ReferenceStatePolicy.receivingStepOrigin",
        result: crate::fea_contract::study_types::classes::FEA_REFERENCE_STATE_POLICY_CLASS,
        signature: "value = fea.ReferenceStatePolicy.receivingStepOrigin()",
        inputs: [],
        output: ("value", "Typed fea.ReferenceStatePolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod explicit_linearization_point {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The transferred state is the point about which the consumer linearizes. `linearize` and `deriveOperatorContribution` require it. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with linearizing transfers.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "explicit-linearization-point",
            "Use explicitLinearizationPoint in a sequence",
            "value = fea.ReferenceStatePolicy.explicitLinearizationPoint();\ntransition = fea.StateTransition.new(\"explicit_linearization_point_transfer\", \"producer\", \"consumer\", \"state\", ...\n    fea.StateTransitionMethod.linearize(), fea.TopologyTransitionPolicy.requireIdentical(), value);",
            "assert(strcmp(class(value), 'fea.ReferenceStatePolicy'));\nassert(strcmp(value.value, 'explicit_linearization_point'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ReferenceStatePolicy.explicitLinearizationPoint",
        summary: "Use the transferred state as an explicit linearization point.",
        description: "`fea.ReferenceStatePolicy.explicitLinearizationPoint()` returns the closed `fea.ReferenceStatePolicy` value `explicit_linearization_point`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "ReferenceStatePolicy",
            "explicitLinearizationPoint",
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
        name: "fea.ReferenceStatePolicy.explicitLinearizationPoint",
        result: crate::fea_contract::study_types::classes::FEA_REFERENCE_STATE_POLICY_CLASS,
        signature: "value = fea.ReferenceStatePolicy.explicitLinearizationPoint()",
        inputs: [],
        output: ("value", "Typed fea.ReferenceStatePolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &preserve::CATALOG_ENTRY,
    &receiving_step_origin::CATALOG_ENTRY,
    &explicit_linearization_point::CATALOG_ENTRY,
];
