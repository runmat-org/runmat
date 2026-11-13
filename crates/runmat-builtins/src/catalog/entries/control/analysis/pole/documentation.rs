use crate::*;
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/pole.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pole"), slug: Some("pole"),
    summary: "Return poles of a transfer-function or state-space model.",
    description: "`pole(sys)` returns denominator roots for a scalar SISO `tf` object or eigenvalues of A for a scalar `ss` object. A static transfer-function gain returns an empty 0-by-1 column.",
    keywords: &["pole", "poles", "stability", "ss", "control system", "roots", "transfer function", "state space"],
    related: &["tf", "ss", "zero"],
    sections: &[
        BuiltinDocumentationSection { heading: "Model semantics", paragraphs: &["The result is an N-by-1 host column. It is real double if all computed poles are real within the control root tolerance, and complex double otherwise. The implementation does not cancel pole-zero pairs. Model-array selectors and model arrays are not implemented; only one model argument is accepted."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Only model fields required for pole extraction are gathered from their owning provider: tf coefficients and metadata, or ss A and Ts. Invalid class and missing required properties reject before downloads. Typed-integer tf coefficients or Ts, logical tf coefficients, and typed-integer consumed ss matrices or Ts assigned after construction use their respective model-property compatibility gates; integer values must convert exactly to the host model domain. This does not add a direct integer sys form. Root and eigenvalue calculations run on CPU; there is no GPU kernel or fusion, and outputs do not alias the model."] },
    ],
    examples: &[
        BuiltinExample { id: "real", title: "Second-order real poles", program: "H = tf(1, [1 3 2]); p = pole(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [2 1])); assert(max(abs(sort(p) - [-2; -1])) < 1e-8);" } },
        BuiltinExample { id: "complex", title: "Complex poles", program: "H = tf(1, [1 0 1]); p = pole(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [2 1])); assert(max(abs(sort(imag(p)) - [-1; 1])) < 1e-8);" } },
        BuiltinExample { id: "static", title: "Static gain", program: "p = pole(tf(2, 1));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [0 1]));" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does pole cancel pole-zero pairs?", answer: "No. It analyzes the stored model denominator or state matrix." }],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        BuiltinDocumentationLink { label: "zero", target: BuiltinDocumentationLinkTarget::Builtin("zero") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "isstable", target: BuiltinDocumentationLinkTarget::Builtin("isstable") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Root classes, shapes, invalid models, and provider preflight", location: "crates/runmat-runtime/src/builtins/control/pole.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflow", location: "crates/runmat-vm/tests/control.rs" }], notes: &["SISO/model-array coverage remains partial. Browser, real provider, and native artifact gates require integration."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
