use crate::*;
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/zero.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("zero"), slug: Some("zero"),
    summary: "Return zeros of a SISO transfer function.",
    description: "`zero(sys)` returns numerator roots of a scalar SISO `tf` object. Static gains return an empty 0-by-1 column. State-space transmission zeros and MIMO models are not yet supported.",
    keywords: &["zero", "zeros", "control system", "roots", "transfer function", "state space"],
    related: &["tf", "pole", "roots"],
    sections: &[
        BuiltinDocumentationSection { heading: "Model semantics", paragraphs: &["The result is a host N-by-1 real double column when all zeros are real within the root tolerance, or a complex double column otherwise. The implementation does not cancel common numerator and denominator factors. Fundamental integer input is not a model and rejects before provider access."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["The parser gathers only required tf properties through the owning provider. Typed-integer or logical coefficients and typed-integer sample time assigned after construction pass the corresponding model-property gate and, for integers, an exact host-model conversion check. This does not admit a direct integer sys. Root computation runs on CPU and is a fusion boundary. No GPU root kernel or implicit upload is used; the output does not alias the input object."] },
    ],
    examples: &[
        BuiltinExample { id: "real", title: "Second-order zeros", program: "H = tf([1 3 2], [1 4]); z = zero(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(z), [2 1])); assert(max(abs(sort(z) - [-2; -1])) < 1e-8);" } },
        BuiltinExample { id: "complex", title: "Complex zeros", program: "H = tf([1 0 1], [1 1]); z = zero(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(z), [2 1])); assert(max(abs(sort(imag(z)) - [-1; 1])) < 1e-8);" } },
        BuiltinExample { id: "static", title: "Static gain", program: "z = zero(tf(5, 2));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(z), [0 1]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does zero cancel pole-zero pairs?", answer: "No. It returns roots of the stored numerator polynomial for the supplied transfer-function model." },
        BuiltinDocumentationFaq { question: "Is zero the same as zeros?", answer: "No. zero analyzes a control model; zeros creates a numeric array." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "pole", target: BuiltinDocumentationLinkTarget::Builtin("pole") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Root classes, shapes, invalid models, and provider preflight", location: "crates/runmat-runtime/src/builtins/control/zero.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflow", location: "crates/runmat-vm/tests/control.rs" }], notes: &["SISO/model-array coverage remains partial. Browser, real provider, and native artifact gates require integration."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
