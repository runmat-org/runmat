use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`cross(A, B)` computes vector cross products for matching 3-element vectors. MATLAB documents single and double data operands; RunMat mode additionally admits logical and all eight real or complex integer classes through explicit compatibility gates. Output is single when either data operand is single and double otherwise.",
            "Inputs `A` and `B` must be the same size.",
            "With no dimension argument, `cross` uses the first dimension whose extent is exactly `3`.",
            "`cross(A, B, dim)` requires `dim` to be a valid array dimension and `size(A, dim) == 3`.",
            "The output has the same shape as the inputs because each input 3-vector produces one output 3-vector.",
            "Typed-integer and logical data operands are RunMat-only extensions, independently gated for A and B, and cross a floating evaluation boundary. Output is single when either data operand is single and double otherwise; arithmetic has floating rounding/overflow behavior rather than integer saturation.",
            "An ordinary positive integer-valued double `dim` is documented. Because the public cross reference does not list typed or logical classes for `dim`, exact all-eight-class typed-integer dimensions and logical dimensions are conservatively classified as separately gated RunMat extensions; resident typed dimensions gather through their owner for exact validation.",
            "A native provider hook is used only for same-owner, same-device real floating handles whose result has the required class. Other supported resident values gather independently and eligible real or complex floating results return to the first owner."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["You usually do not need to call `gpuArray` explicitly for `cross`. RunMat uses a class-preserving native hook for eligible same-owner real inputs and otherwise follows a gather -> host compute -> class-preserving re-upload path for supported real and complex floating values.", "The WGPU provider computes real-valued `cross` products by gathering the three vector components into temporary device tensors, applying multiply/subtract kernels on-device, and scattering the results back into the output tensor. Providers that do not implement `cross` fall back to the host reference path.", "Same-owner, same-device real floating inputs may use a provider `cross` hook when its result preserves the required f32/f64 class and belongs to the invoked provider/device. Other resident inputs gather through their owners; eligible real or complex floating results are restored when the owner preserves class and complexity."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "example-1", title: "Computing the cross product of row vectors", program: "a = [1 0 0];\nb = [0 1 0];\nc = cross(a, b)", display_output: Some("c =\n     0     0     1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(c, [0 0 1]));" } },
    BuiltinExample { id: "example-2", title: "Computing the cross product of column vectors", program: "u = [1; 0; 0];\nv = [0; 1; 0];\nw = cross(u, v)", display_output: Some("w =\n     0\n     0\n     1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(w, [0; 0; 1]));" } },
    BuiltinExample { id: "example-3", title: "Applying `cross` row-wise across a matrix", program: "A = [1 0 0; 0 1 0];\nB = [0 1 0; 0 0 1];\nC = cross(A, B, 2)", display_output: Some("C =\n     0     0     1\n     1     0     0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(C, [0 0 1; 1 0 0]));" } },
    BuiltinExample { id: "example-4", title: "Evaluating `cross` on GPU-resident tensors", program: "G1 = gpuArray([1 0 0]);\nG2 = gpuArray([0 1 0]);\nG = cross(G1, G2);\nresult = gather(G)", display_output: Some("result =\n     0     0     1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(result, [0 0 1]));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `cross` require vectors of length 3?", answer: "Yes. The operating dimension must have extent `3`, whether the vectors are stored as rows, columns, or slices of a higher-rank tensor." },
    BuiltinDocumentationFaq { question: "How is the default dimension chosen?", answer: "RunMat follows MATLAB semantics and picks the first dimension whose size is exactly `3`." },
    BuiltinDocumentationFaq { question: "What happens if I pass `dim` explicitly?", answer: "The dimension must exist and must have length `3`; otherwise `cross` raises a descriptive error." },
    BuiltinDocumentationFaq { question: "Can I use complex inputs?", answer: "Yes for documented single/double complex inputs. RunMat mode also accepts typed complex integers under the corresponding operand gate, converts both components to double, and applies the standard bilinear formula without conjugation." },
    BuiltinDocumentationFaq { question: "Does the result stay on the GPU?", answer: "Same-owner, same-device real floating inputs remain resident when a provider's `cross` hook returns the required class. Other resident inputs gather independently, and eligible real or complex floating results return to the first owner. Typed complex integers work on the host, but the current provider ABI cannot encode typed complex-integer resident buffers." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "MATLAB cross documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/cross.html") },
    BuiltinDocumentationLink { label: "dot", target: BuiltinDocumentationLinkTarget::Builtin("dot") },
    BuiltinDocumentationLink { label: "kron", target: BuiltinDocumentationLinkTarget::Builtin("kron") },
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "norm", target: BuiltinDocumentationLinkTarget::Builtin("norm") },
    BuiltinDocumentationLink { label: "sum", target: BuiltinDocumentationLinkTarget::Builtin("sum") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/cross.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cross"),
    slug: Some("cross"),
    summary: "Compute vector cross products in MATLAB and RunMat.",
    description: "`cross(A, B)` computes vector cross products for matching 3-element vectors. MATLAB documents single and double data operands; RunMat mode additionally admits logical and all eight real or complex integer classes through explicit compatibility gates. Output is single when either data operand is single and double otherwise.",
    keywords: &["cross", "cross product", "vector product", "3d vector", "gpu"],
    related: &["dot", "kron", "mtimes", "norm", "sum"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/cross.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::ops::cross::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::ops::cross::tests::cross_gpu_roundtrip" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Existing GPU parity tests", location: "builtins::math::linalg::ops::cross::tests::cross_wgpu_matches_cpu" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
