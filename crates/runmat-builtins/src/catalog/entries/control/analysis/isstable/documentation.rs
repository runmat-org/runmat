use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/isstable.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isstable"),
    slug: Some("isstable"),
    summary: "Test stability of a SISO transfer-function model.",
    description: "`isstable(sys)` returns a logical scalar after calculating the poles of a SISO transfer-function model. Continuous-time poles must lie in the open left half-plane; discrete-time poles must lie inside the unit circle. The numerical test uses the control model's pole tolerance near either boundary.",
    keywords: &["isstable", "stability", "poles", "control system", "tf"],
    related: &["pole", "tf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Model and stability rules", paragraphs: &[
            "The currently supported input is a scalar SISO tf object. A transfer function with no dynamic poles, such as a static gain, is stable. An invalid transfer-function object or unsupported model class raises an error instead of returning false.",
            "For continuous-time models, the pole test requires each real component to be less than -1e-12. For discrete-time models, each pole magnitude must be less than 1-1e-12. A pole on or very near the imaginary axis or unit circle therefore returns false. These numerical margins are current RunMat behavior; exact MATLAB boundary behavior has not been established here.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "The result is always a host logical scalar. Typed-integer tf coefficients or Ts and logical tf coefficients assigned after construction are admitted through their model-property compatibility gates, and integer values must convert exactly to the host model domain. Transfer-function coefficient values stored on an acceleration provider are gathered through their owner while the model is parsed; pole analysis then runs on the host. This operation has no GPU kernel and is a fusion boundary.",
            "The call may suspend while gathering nested resident coefficient values and may throw for an invalid model or failed pole calculation. It has no callback, workspace, filesystem, network, user-interface, randomness, or time effect. Remote execution-owned model objects are not supported.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "stable-continuous", title: "Stable continuous-time model", program: "H = tf(1, [1 1]); stable = isstable(H); fprintf('%d\\n', stable);", display_output: Some("1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(stable)); assert(stable);" } },
        BuiltinExample { id: "unstable-and-marginal", title: "Unstable and marginal continuous-time poles", program: "unstable = isstable(tf(1, [1 -1])); marginal = isstable(tf(1, [1 0]));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~unstable); assert(~marginal);" } },
        BuiltinExample { id: "discrete-and-static", title: "Stable discrete-time model and static gain", program: "D = tf(1, [1 -0.5], 1); d = isstable(D); g = isstable(tf(2, 1));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(d); assert(g);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does marginal stability return true?", answer: "No. Poles on or within the current numerical tolerance of the imaginary axis or unit circle are not classified as stable." },
        BuiltinDocumentationFaq { question: "Can isstable analyze a state-space model?", answer: "Not yet. The current implementation accepts SISO tf objects; other model classes produce an unsupported-model error." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pole", target: BuiltinDocumentationLinkTarget::Builtin("pole") },
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Transfer-function stability and invalid numeric inputs", location: "crates/runmat-runtime/src/builtins/control/isstable.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM transfer-function workflow", location: "crates/runmat-vm/tests/control.rs" },
        ],
        notes: &["The SISO tf-only model domain and numerical boundary margin are explicit partial-coverage limits. Native artifact, browser, and provider behavior require integrated qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
