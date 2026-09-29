use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/ss.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ss"), slug: Some("ss"),
    summary: "Create a state-space model from A, B, C, and D matrices.",
    description: "`ss(A,B,C,D)` constructs a host `ss` object. A must be square; B has the same row count as A; C has the same column count as A; and D has C's row count and B's column count. Inputs must be finite real numeric scalars or dense matrices.",
    keywords: &["ss", "state space", "control system", "state matrix", "sample time"],
    related: &["tf", "step", "impulse"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and stored model", paragraphs: &[
            "A scalar is treated as a 1-by-1 matrix. The returned object retains each input matrix's orientation after normalization to double. It stores A, B, C, D, Ts, zero InputDelay and OutputDelay vectors, and empty StateName, InputName and OutputName cell arrays. Without Ts it is continuous-time with Ts = 0.",
            "`ss(A,B,C,D,Ts)` and the `Ts` or `SampleTime` name/value forms accept a finite nonnegative scalar sample time. Option names are case-insensitive. Complex, sparse, descriptor-form, uncertain, identified, and generalized models are outside this implementation.",
            "Typed-integer matrices and sample times are RunMat-only extensions. Values must cross the binary64 model boundary exactly. An inexact matrix reports RunMat:ss:InvalidMatrix; an inexact sample time reports RunMat:ss:InvalidSampleTime. Explicit gpuArray input is a separate RunMat-only extension; ordinary automatically resident inputs may gather in either mode.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Extension admission precedes provider access. Supported resident matrices and sample times then gather through their owning provider and are checked for exact integer conversion at the host model boundary. Construction validates dimensions and returns a new value-semantic host object. Host tensor storage may be shared until copy-on-write mutation. There is no GPU kernel or fusion; provider access may suspend.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "stored-matrices", title: "Read stored matrices in column-major order", program: "G = ss([0 1; -2 -3], [0; 1], [1 0], 0);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(G.A(:), [0; -2; 1; -3])); assert(isequal(G.B, [0; 1])); assert(isequal(G.C, [1 0])); assert(G.D == 0);" } },
        BuiltinExample { id: "continuous", title: "Continuous state-space model", program: "A = [0 1; -2 -3]; B = [0; 1]; C = [1 0]; G = ss(A,B,C,0);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(G), 'ss')); assert(isequal(G.A, A)); assert(isequal(size(G.B), [2 1])); assert(G.Ts == 0);" } },
        BuiltinExample { id: "discrete", title: "Discrete sample time", program: "G = ss(0.5,1,1,0,0.1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(G.Ts - 0.1) < 1e-12); assert(isequal(size(G.A), [1 1]));" } },
        BuiltinExample { id: "named-sample-time", title: "Named sample time", program: "G = ss(0.5,1,1,0,'SampleTime',0.2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(G.Ts - 0.2) < 1e-12);" } },
        BuiltinExample { id: "integer-extension", title: "Integer matrix extension", program: "G = ss(int16(1),int16(2),int16(3),int16(4));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(G.A), 'double')); assert(G.A == 1); assert(G.D == 4);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How are matrix dimensions validated?", answer: "A must be n-by-n, B n-by-nu, C ny-by-n, and D ny-by-nu. Their stored orientations are preserved." },
        BuiltinDocumentationFaq { question: "Does ss simulate a response?", answer: "No. It constructs a model; step and impulse perform response calculations." },
        BuiltinDocumentationFaq { question: "Are complex and sparse matrices supported?", answer: "No. This constructor currently requires finite real dense matrices." },
        BuiltinDocumentationFaq { question: "Is the result GPU-resident?", answer: "No. Accepted resident inputs are gathered and the model owns host-side double matrices." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "step", target: BuiltinDocumentationLinkTarget::Builtin("step") },
        BuiltinDocumentationLink { label: "impulse", target: BuiltinDocumentationLinkTarget::Builtin("impulse") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Constructor forms, dimensions, compatibility, and resident inputs", location: "crates/runmat-runtime/src/builtins/control/ss.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["MIMO matrix storage is supported, but descriptor, sparse, complex, and generalized models are not. Browser, provider and native-artifact tests remain integrated gates."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
