use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &[
            "`B = ctranspose(A)` (or `A'`) swaps the first two dimensions and conjugates complex values. Real-valued input behaves like simple transpose. Scalars, vectors, matrices, and explicit trailing singleton dimensions are supported; N-D arrays with a nonsingleton dimension above two are rejected in favor of `permute`.",
            "Real numeric, logical, and character entries are unchanged by conjugation. Complex single and double entries are conjugated and remain complex even when every imaginary component is zero. Complex integer storage is rejected. Empty arrays use column-major transpose semantics.",
            "Dense real integer input retains all eight integer classes. Integer-valued sparse storage requires the `ctranspose-integer-sparse` RunMat extension. Character and cell arrays retain their types while entries move to transposed positions. String scalars pass through and string arrays transpose. Scalar structs are unchanged; struct arrays transpose. External objects may provide their own `ctranspose` method.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &[
            "Real and complex provider-resident tensors use the exact provider's transpose or permute operation and, for complex input, its `unary_conj` hook when available. Single and double are the supported floating precisions; no broadcast or reduction is performed.",
            "When a required hook is unavailable, RunMat gathers to host, conjugate-transposes, and attempts to re-upload. The result may remain on host if re-upload is unavailable. Complex fallback preserves interleaved storage metadata when the result remains complex. A partial transpose/conjugation hook failure selects the host fallback and emits a warning. The operation is a fusion boundary so downstream kernels see the new shape.",
        ],
    },
];

const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: Option<&'static str>,
    harness: BuiltinExampleHarness,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "complex-matrix",
        "Conjugate transpose of a complex matrix",
        "Z = [1+2i 3-4i; 5+0i 6-7i];\nH = ctranspose(Z)",
        Some("H =\n   1 - 2i   5 - 0i\n   3 + 4i   6 + 7i"),
        BuiltinExampleHarness::Portable,
        "assert(isequal(H, [1-2i 5-0i; 3+4i 6+7i]));",
    ),
    example(
        "real-matrix",
        "Conjugate transpose of a real matrix equals the plain transpose",
        "A = [1 2 3; 4 5 6];\nB = ctranspose(A)",
        Some("B =\n     1     4\n     2     5\n     3     6"),
        BuiltinExampleHarness::Portable,
        "assert(isequal(B, [1 4; 2 5; 3 6]));",
    ),
    example(
        "row-vector",
        "Conjugate transpose turns row vectors into column vectors",
        "row = [1-2i, 3+4i, 5];\ncol = ctranspose(row);\nsize(col)",
        Some("ans = [3 1]"),
        BuiltinExampleHarness::Portable,
        "assert(isequal(size(col), [3 1]));",
    ),
    example(
        "complex-scalar",
        "Conjugate transpose of a complex scalar",
        "z = 2 + 3i;\nresult = ctranspose(z)",
        Some("result = 2 - 3i"),
        BuiltinExampleHarness::Portable,
        "assert(result == 2 - 3i);",
    ),
    example(
        "characters",
        "Conjugate transpose of text data preserves characters",
        "C = ['r' 'u' 'n'; 'm' 'a' 't'];\nCT = ctranspose(C)",
        Some("CT =\n    'rm'\n    'ua'\n    'nt'"),
        BuiltinExampleHarness::Portable,
        "assert(isequal(size(CT), [3 2]));",
    ),
    example(
        "resident-complex",
        "Conjugate transpose of a gpuArray",
        "G = gpuArray(rand(1024, 64) + 1i * rand(1024, 64));\nGT = ctranspose(G)",
        None,
        BuiltinExampleHarness::Wgpu,
        "assert(isequal(size(GT), [64 1024]));",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `ctranspose` different from `transpose`?", answer: "`transpose` (`A.'`) swaps dimensions without conjugation; `ctranspose` (`A'`) also conjugates complex values. For purely real data they are identical." },
    BuiltinDocumentationFaq { question: "Does `ctranspose` change logical or character arrays?", answer: "Only their layout changes. Values remain logical or character, and conjugation has no effect." },
    BuiltinDocumentationFaq { question: "What about higher-dimensional arrays?", answer: "Use `permute` for higher-dimensional arrays. `ctranspose` accepts vectors and matrices, including explicit trailing singleton dimensions, and rejects nonsingleton dimensions above two." },
    BuiltinDocumentationFaq { question: "Does the result share storage with the input?", answer: "An array transpose usually materializes a new arrangement. A provider may return a lazy view or alias for an identity-shaped result, so storage independence is not guaranteed; value semantics remain unchanged." },
    BuiltinDocumentationFaq { question: "How are complex tensors handled on the GPU today?", answer: "Complex gpuArray values use interleaved real/imaginary storage. Providers with `transpose`/`permute` and `unary_conj` hooks keep the conjugate transpose resident; fallback gathers and attempts to re-upload through the same complex storage metadata." },
    BuiltinDocumentationFaq { question: "Will `ctranspose` fuse with neighbouring kernels?", answer: "Conjugate transposes currently act as fusion boundaries so that shape changes are visible to downstream kernels." },
    BuiltinDocumentationFaq { question: "Can I rely on `ctranspose` inside linear-algebra routines (e.g., Hermitian products)?", answer: "Yes for the supported numeric, logical, character, string, cell, scalar-struct, sparse, and resident array forms described above. On those inputs it preserves the conjugate-transpose semantics required by idioms such as `A' * A`." },
    BuiltinDocumentationFaq { question: "What error do I get for unsupported types?", answer: "Supported scalar structs are unchanged by transpose. Unsupported value kinds raise `ctranspose: unsupported input type ...`; external objects can provide their own `ctranspose` method." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "transpose", target: BuiltinDocumentationLinkTarget::Builtin("transpose") },
    BuiltinDocumentationLink { label: "conj", target: BuiltinDocumentationLinkTarget::Builtin("conj") },
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "dot", target: BuiltinDocumentationLinkTarget::Builtin("dot") },
    BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
    BuiltinDocumentationLink { label: "mpower", target: BuiltinDocumentationLinkTarget::Builtin("mpower") },
    BuiltinDocumentationLink { label: "mrdivide", target: BuiltinDocumentationLinkTarget::Builtin("mrdivide") },
    BuiltinDocumentationLink { label: "trace", target: BuiltinDocumentationLinkTarget::Builtin("trace") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/ctranspose.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ctranspose"),
    slug: Some("ctranspose"),
    summary: "Conjugate-transpose arrays in MATLAB and RunMat.",
    description: "`B = ctranspose(A)` (or `A'`) swaps the first two dimensions of `A` and conjugates complex values. Real-valued inputs therefore behave like simple transpose, with MATLAB/RunMat Hermitian semantics for complex data.",
    keywords: &["ctranspose", "conjugate transpose", "hermitian", "gpu", "matrix transpose"],
    related: &["transpose", "conj", "mtimes", "gpuArray", "gather", "dot", "mldivide", "mpower", "mrdivide", "trace"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Conjugate-transpose runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/ctranspose.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, integer, object, and provider transpose tests", location: "builtins::math::linalg::ops::ctranspose::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Provider transpose round trip", location: "builtins::math::linalg::ops::ctranspose::tests::ctranspose_gpu_roundtrip" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "WGPU and host numerical parity", location: "builtins::math::linalg::ops::ctranspose::tests::ctranspose_wgpu_matches_cpu" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
