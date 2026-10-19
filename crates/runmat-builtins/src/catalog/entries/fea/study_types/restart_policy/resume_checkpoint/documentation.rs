use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Both arguments are non-empty text. A checkpoint id alone is insufficient: the digest stops a checkpoint from a different model, sequence, topology, or state from being accepted.",
        "Model validation allows checkpoint resume only for transient, nonlinear, or adaptive procedures.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `restart` argument (8th) of `fea.ProcedureStep.configure`.",
        "Direct and modal transient continuation resume from a complete integrator-state checkpoint; stateless fan-out work is replayed rather than resumed.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the checkpoint id or digest is empty.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "resume",
        title: "Resume a transient from checkpoint 42",
        program: "policy = fea.RestartPolicy.resumeCheckpoint(\"transient_ckpt_0042\", \"sha256:9f2c4e\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.RestartPolicy'));\nassert(strcmp(policy.value.resume_checkpoint.checkpoint_id, 'transient_ckpt_0042'));\nassert(strcmp(policy.value.resume_checkpoint.required_state_digest, 'sha256:9f2c4e'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RestartPolicy.resumeCheckpoint",
        summary: "Resume a step only from a digest-fenced checkpoint.",
        description: "`fea.RestartPolicy.resumeCheckpoint(checkpointId, requiredStateDigest)` builds a `fea.RestartPolicy` that resumes a step from the named checkpoint only when its state digest matches.",
        keywords: &["fea", "sequence", "restart", "checkpoint", "resume", "digest"],
        related: &["fea.RestartPolicy.fresh", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can a static step resume a checkpoint?", answer: "No. Only transient, nonlinear, or adaptive procedures may resume a checkpoint." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/restart_policy/tests.rs::restart_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Checkpoint restart identity", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_checkpoint_restart_without_stable_identity" },
        ],
        notes: &[],
    });
