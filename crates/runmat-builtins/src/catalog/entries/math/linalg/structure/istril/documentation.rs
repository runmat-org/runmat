use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`istril(A)` returns true when every element above the main diagonal of matrix `A` is zero.", "Scalars are lower triangular.", "Diagonal and all-zero matrices are lower triangular.", "Rectangular matrices are supported.", "Dense real, logical, complex, sparse, and gpuArray inputs are supported.", "Complex entries count as nonzero when either real or imaginary part is nonzero.", "NaN values count as nonzero when testing matrix structure.", "Arrays with a non-singleton dimension beyond the second return `false`, matching MATLAB matrix-predicate behavior."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["The result is always a host logical scalar. For real/logical gpuArray inputs, RunMat keeps the matrix on the provider and uses the `bandwidth` hook to read back only lower/upper bandwidth metadata. Complex-interleaved GPU inputs and providers without `bandwidth` support fall back to the host path.", "`istril` is registered as a structure-analysis fusion sink. GPU-resident real/logical inputs use provider bandwidth (`upper == 0`) instead of gathering the full matrix when the active provider supports it.", "Uses the provider `bandwidth` hook for real/logical gpuArray inputs and reads back only the small bandwidth result; complex/provider-missing paths fall back to the host."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Checking a lower-triangular matrix",
        program: "A = [1 0 0; 2 3 0; 4 5 6];\ntf = istril(A)",
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
        title: "Detecting a value above the main diagonal",
        program: "B = [1 7; 0 2];\ntf = istril(B)",
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
        title: "Checking a rectangular matrix",
        program: "C = [1 0; 2 3; 4 5];\ntf = istril(C)",
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
        program: "G = gpuArray(tril(ones(3)));\ntf = istril(G)",
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
    BuiltinDocumentationFaq { question: "What entries does `istril` inspect?", answer: "Only entries above the main diagonal can make the predicate false." },
    BuiltinDocumentationFaq { question: "Are diagonal matrices lower triangular?", answer: "Yes. Diagonal matrices have zeros above the main diagonal, so `istril` returns true." },
    BuiltinDocumentationFaq { question: "How are sparse matrices handled?", answer: "RunMat scans stored sparse entries without densifying and returns false only when a stored nonzero appears above the main diagonal." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isdiag", target: BuiltinDocumentationLinkTarget::Builtin("isdiag") },
    BuiltinDocumentationLink { label: "istriu", target: BuiltinDocumentationLinkTarget::Builtin("istriu") },
    BuiltinDocumentationLink { label: "tril", target: BuiltinDocumentationLinkTarget::Builtin("tril") },
    BuiltinDocumentationLink { label: "bandwidth", target: BuiltinDocumentationLinkTarget::Builtin("bandwidth") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/diagonal_triangular.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("istril"),
    slug: Some("istril"),
    summary: "Determine whether a matrix is lower triangular in MATLAB and RunMat.",
    description: "`istril(A)` returns true when every element above the main diagonal of matrix `A` is zero.",
    keywords: &["istril", "lower triangular", "matrix structure", "sparse", "gpu"],
    related: &["isdiag", "istriu", "tril", "bandwidth", "gpuarray", "gather"],
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
