//! `fea.TopologyTransitionPolicy` choices.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod require_identical {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The transfer is valid only when both steps use the same mesh topology; RunMat checks the topology digests before transferring. `reuse` requires this policy, and it is the only topology policy RunMat executes today.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for transfers within one mesh.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "require-identical",
            "Use requireIdentical in a sequence",
            "value = fea.TopologyTransitionPolicy.requireIdentical();\ntransition = fea.StateTransition.new(\"require_identical_transfer\", \"producer\", \"consumer\", \"state\", ...\n    fea.StateTransitionMethod.reuse(), value, fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.TopologyTransitionPolicy'));\nassert(strcmp(value.value, 'require_identical'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TopologyTransitionPolicy.requireIdentical",
        summary: "Require producer and consumer to share identical topology.",
        description: "`fea.TopologyTransitionPolicy.requireIdentical()` returns the closed `fea.TopologyTransitionPolicy` value `require_identical`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "TopologyTransitionPolicy",
            "requireIdentical",
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
        name: "fea.TopologyTransitionPolicy.requireIdentical",
        result: crate::fea_contract::study_types::classes::FEA_TOPOLOGY_TRANSITION_POLICY_CLASS,
        signature: "value = fea.TopologyTransitionPolicy.requireIdentical()",
        inputs: [],
        output: ("value", "Typed fea.TopologyTransitionPolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod project_with_evidence {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer may use a different mesh as long as the transfer carries evidence of its accuracy. Execution does not perform topology-changing transfers yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare transfers between independently meshed steps.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "project-with-evidence",
            "Use projectWithEvidence in a sequence",
            "value = fea.TopologyTransitionPolicy.projectWithEvidence();\ntransition = fea.StateTransition.new(\"project_with_evidence_transfer\", \"producer\", \"consumer\", \"state\", ...\n    fea.StateTransitionMethod.interpolation(), value, fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.TopologyTransitionPolicy'));\nassert(strcmp(value.value, 'project_with_evidence'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TopologyTransitionPolicy.projectWithEvidence",
        summary: "Permit transfer between different topologies with projection evidence.",
        description: "`fea.TopologyTransitionPolicy.projectWithEvidence()` returns the closed `fea.TopologyTransitionPolicy` value `project_with_evidence`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "TopologyTransitionPolicy",
            "projectWithEvidence",
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
        name: "fea.TopologyTransitionPolicy.projectWithEvidence",
        result: crate::fea_contract::study_types::classes::FEA_TOPOLOGY_TRANSITION_POLICY_CLASS,
        signature: "value = fea.TopologyTransitionPolicy.projectWithEvidence()",
        inputs: [],
        output: ("value", "Typed fea.TopologyTransitionPolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod adapted_mesh_lineage {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "The consumer's mesh must be an adaptation of the producer's, and the method must be conservative projection or interpolation. Execution does not perform it yet; a study using it fails when it runs.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to declare transfers across adaptive refinement.",
            ],
        ),
        USE_TRANSITION,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "adapted-mesh-lineage",
            "Use adaptedMeshLineage in a sequence",
            "value = fea.TopologyTransitionPolicy.adaptedMeshLineage();\ntransition = fea.StateTransition.new(\"adapted_mesh_lineage_transfer\", \"producer\", \"consumer\", \"state\", ...\n    fea.StateTransitionMethod.conservativeProjection(), value, fea.ReferenceStatePolicy.preserve());",
            "assert(strcmp(class(value), 'fea.TopologyTransitionPolicy'));\nassert(strcmp(value.value, 'adapted_mesh_lineage'));\nassert(strcmp(class(transition), 'fea.StateTransition'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateTransition.new",
        "fea.SequenceDependencyReason.stateTransfer",
        "fea.AnalysisSequence.new",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TopologyTransitionPolicy.adaptedMeshLineage",
        summary: "Require the consumer mesh to descend from the producer mesh by adaptation.",
        description: "`fea.TopologyTransitionPolicy.adaptedMeshLineage()` returns the closed `fea.TopologyTransitionPolicy` value `adapted_mesh_lineage`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "TopologyTransitionPolicy",
            "adaptedMeshLineage",
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
        name: "fea.TopologyTransitionPolicy.adaptedMeshLineage",
        result: crate::fea_contract::study_types::classes::FEA_TOPOLOGY_TRANSITION_POLICY_CLASS,
        signature: "value = fea.TopologyTransitionPolicy.adaptedMeshLineage()",
        inputs: [],
        output: ("value", "Typed fea.TopologyTransitionPolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &require_identical::CATALOG_ENTRY,
    &project_with_evidence::CATALOG_ENTRY,
    &adapted_mesh_lineage::CATALOG_ENTRY,
];
