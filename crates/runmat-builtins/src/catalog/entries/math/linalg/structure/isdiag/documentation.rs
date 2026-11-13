use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`isdiag(A)` returns true when every nonzero element of matrix `A` lies on the main diagonal.", "Scalars are diagonal.", "Rectangular matrices can be diagonal when every off-diagonal entry is zero.", "All-zero and empty matrices are diagonal.", "Dense real, logical, complex, sparse, and gpuArray inputs are supported.", "Complex entries count as nonzero when either real or imaginary part is nonzero.", "NaN values count as nonzero when testing matrix structure.", "Arrays with a non-singleton dimension beyond the second return `false`, matching MATLAB matrix-predicate behavior."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["The result is always a host logical scalar. For real/logical gpuArray inputs, RunMat keeps the matrix on the provider and uses the `bandwidth` hook to read back only lower/upper bandwidth metadata. Complex-interleaved GPU inputs and providers without `bandwidth` support fall back to the host path.", "`isdiag` is registered as a structure-analysis fusion sink. GPU-resident real/logical inputs use provider bandwidth (`lower == 0 && upper == 0`) instead of gathering the full matrix when the active provider supports it.", "Uses the provider `bandwidth` hook for real/logical gpuArray inputs and reads back only the small bandwidth result; complex/provider-missing paths fall back to the host."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Checking a diagonal matrix",
        program: "A = [4 0 0; 0 5 0; 0 0 6];\ntf = isdiag(A)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Detecting an off-diagonal value",
        program: "B = [1 0; 7 2];\ntf = isdiag(B)",
        display_output: Some("tf = logical\n   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Checking a rectangular diagonal matrix",
        program: "C = [1 0; 0 2; 0 0];\ntf = isdiag(C)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Inspecting a GPU-resident matrix",
        program: "G = gpuArray(eye(3));\ntf = isdiag(G)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `isdiag` require a square matrix?", answer: "No. A rectangular matrix is diagonal when all entries off the main diagonal are zero." },
    BuiltinDocumentationFaq { question: "How are sparse matrices handled?", answer: "RunMat scans stored sparse entries without densifying and returns false only when a stored nonzero appears off the main diagonal." },
    BuiltinDocumentationFaq { question: "Does `NaN` count as zero?", answer: "No. A NaN entry is treated as nonzero for structure predicates." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "istril", target: BuiltinDocumentationLinkTarget::Builtin("istril") },
    BuiltinDocumentationLink { label: "istriu", target: BuiltinDocumentationLinkTarget::Builtin("istriu") },
    BuiltinDocumentationLink { label: "bandwidth", target: BuiltinDocumentationLinkTarget::Builtin("bandwidth") },
    BuiltinDocumentationLink { label: "diag", target: BuiltinDocumentationLinkTarget::Builtin("diag") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/diagonal_triangular.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isdiag"),
    slug: Some("isdiag"),
    summary: "Determine whether a matrix is diagonal in MATLAB and RunMat.",
    description: "`isdiag(A)` returns true when every nonzero element of matrix `A` lies on the main diagonal.",
    keywords: &["isdiag", "diagonal matrix", "matrix structure", "sparse", "gpu"],
    related: &["istril", "istriu", "bandwidth", "diag", "gpuarray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/diagonal_triangular.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::structure::diagonal_triangular::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::structure::diagonal_triangular::tests::gpu_inputs_use_provider_bandwidth_without_gathering" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
