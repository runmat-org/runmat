use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`pagemtimes(X,Y)` multiplies each corresponding two-dimensional matrix page of `X` and `Y`. Dimensions three and higher are page dimensions and use MATLAB-compatible singleton expansion.",
            "Supports `Z = pagemtimes(X,Y)` and `Z = pagemtimes(X,transpX,Y,transpY)`.",
            "`transpX` and `transpY` must both be present in the four-argument form and each must be `none`, `transpose`, or `ctranspose`.",
            "The first two dimensions of each input form the matrix multiplied on each page. Page dimensions are dimensions three and higher and are expanded by position when one side has extent `1` or omits the trailing dimension.",
            "If either operand is scalar, the result uses the first two dimensions of the nonscalar operand and performs page-wise scalar multiplication.",
            "Dense `double`, `single`, and complex double arrays are supported. MATLAB-compatible modes reject integer, logical, sparse, cell, struct, object, and text inputs.",
            "RunMat mode additionally accepts real arrays in all eight integer classes when every value is exactly representable as binary64. This named extension returns double and rejects wider values instead of rounding them; resident integer inputs use an exact owner-aware gather and restore path rather than a floating provider kernel.",
            "Complex `ctranspose` conjugates each page before multiplication; `transpose` only swaps the first two dimensions."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["Yes for explicit `gpuArray` inputs. RunMat gathers when needed for host-complete semantics, then re-uploads real and complex numeric results to the active acceleration provider.", "`pagemtimes` exposes a custom GPU spec and keeps outputs resident when any input is a `gpuArray` and an acceleration provider is available.", "Plain page-wise products may be accelerated by provider `pagefun` hooks in future backends; transpose and scalar forms currently use the host-complete fallback path.", "RunMat gathers GPU operands for host-complete scalar and transpose semantics, then uploads real and complex numeric results back to the active provider."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Multiply matching pages",
        program: "X = reshape(1:8, [2 2 2]);\nY = reshape(10:17, [2 2 2]);\nZ = pagemtimes(X,Y)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isequal(size(Z), [2 2 2])); assert(isequal(Z(:,:,1), X(:,:,1)*Y(:,:,1)));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Broadcast a matrix across all pages",
        program: "X = eye(2);\nY = reshape(1:8, [2 2 2]);\nZ = pagemtimes(X,Y)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Z), [2 2 2])); assert(isequal(Z(:,:,2), Y(:,:,2)));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Use transpose options",
        program:
            "X = rand(2,3,5);\nY = rand(2,4,5);\nZ = pagemtimes(X,'transpose',Y,'none');\nsize(Z)",
        display_output: Some("ans =\n     3     4     5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Z), [3 4 5]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Implicit expansion of page dimensions",
        program: "X = rand(10,8,1,3);\nY = rand(8,10,4,1);\nZ = pagemtimes(X,Y);\nsize(Z)",
        display_output: Some("ans =\n    10    10     4     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Z), [10 10 4 3]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "GPU inputs remain device-resident",
        program: "G = gpuArray(rand(3,3,8));\nH = pagemtimes(G,2);\nfirstPage = gather(H(:,:,1));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(firstPage, 2*gather(G(:,:,1))));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How does `pagemtimes` differ from `mtimes`?", answer: "`mtimes` multiplies two matrices. `pagemtimes` treats every slice in dimensions three and higher as an independent matrix and multiplies corresponding pages with singleton expansion across page dimensions." },
    BuiltinDocumentationFaq { question: "Can I specify only one transpose option?", answer: "No. MATLAB's four-argument form requires both `transpX` and `transpY`; use `none` for the operand that should not be transposed." },
    BuiltinDocumentationFaq { question: "What happens when page dimensions do not match?", answer: "RunMat raises `RunMat:pagemtimes:PageDimensionMismatch` unless the differing page extent is `1` or omitted on one side." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "transpose", target: BuiltinDocumentationLinkTarget::Builtin("transpose") },
    BuiltinDocumentationLink { label: "ctranspose", target: BuiltinDocumentationLinkTarget::Builtin("ctranspose") },
    BuiltinDocumentationLink { label: "pagefun", target: BuiltinDocumentationLinkTarget::Builtin("pagefun") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/pagemtimes.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pagemtimes"),
    slug: Some("pagemtimes"),
    summary: "Multiply corresponding matrix pages in dense arrays.",
    description: "`pagemtimes(X,Y)` multiplies each corresponding two-dimensional matrix page of `X` and `Y`. Dimensions three and higher are page dimensions and use MATLAB-compatible singleton expansion.",
    keywords: &["pagemtimes", "page-wise matrix multiplication", "batched matrix multiply", "linear algebra", "gpuArray"],
    related: &["mtimes", "transpose", "ctranspose", "pagefun", "gpuarray"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/pagemtimes.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::ops::pagemtimes::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::ops::pagemtimes::tests::gpu_input_roundtrips_to_gpu_result" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
