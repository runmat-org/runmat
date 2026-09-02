use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Values and classes",
        paragraphs: &[
            "`allfinite(A)` returns one logical scalar. The result is true only when every real value, and both components of every complex value, are finite. `NaN`, positive infinity, and negative infinity make the result false.",
            "All fixed-width integers, logical values, and character code points are finite. Sparse input checks stored values; implicit zeros are finite. Empty numeric arrays return true, following the identity of an `all` reduction.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Strings and unsupported containers",
        paragraphs: &[
            "RunMat compatibility mode additionally accepts strings. A string scalar or nonempty string array returns false; an empty string array returns true. MATLAB compatibility mode rejects this extension.",
            "Cells, structs, objects, function handles, and other nonnumeric containers return `RunMat:allfinite:InvalidInput`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &[
            "A resident floating input uses the exact owning provider's finite-classification and logical-reduction operations when both are available. The scalar result is returned on the host. A typed unsupported result falls back through one class-preserving input transfer; other provider failures remain errors.",
            "Resident integer and logical inputs return true from validated class metadata without downloading payload data.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "finite-matrix", title: "Check a finite matrix", program: "tf = allfinite([1 2; 3 4])", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "nonfinite", title: "Detect NaN and infinity", program: "tf = allfinite([1 NaN Inf])", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "complex", title: "Check both complex components", program: "finite = allfinite([1+2i 3-4i]);\nnonfinite = allfinite([1+2i complex(3, Inf)])", display_output: Some("finite = true and nonfinite = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(finite);\nassert(~nonfinite);" } },
    BuiltinExample { id: "empty", title: "Reduce an empty numeric array", program: "tf = allfinite(zeros(0, 3))", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "integers", title: "Check wide integers without conversion", program: "A = [uint64(0), uint64(9007199254740992) + uint64(1), intmax('uint64')];\ntf = allfinite(A)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "string-extension", title: "Use the RunMat string extension", program: "tf = allfinite(\"data\")", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "gpu-reduction", title: "Reduce a resident array to a host scalar", program: "G = gpuArray(single([1 2 Inf]));\ntf = allfinite(G)", display_output: Some("tf is a false host logical scalar"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);\nassert(~isgpuarray(tf));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `allfinite(A)` different from `isfinite(A)`?", answer: "`isfinite` returns a same-shaped logical mask. `allfinite` reduces the whole input to one logical scalar." },
    BuiltinDocumentationFaq { question: "Are empty arrays all finite?", answer: "Yes. An empty finite mask reduces to true." },
    BuiltinDocumentationFaq { question: "How are complex values checked?", answer: "Every real and imaginary component must be finite." },
    BuiltinDocumentationFaq { question: "Are integer values converted to floating point?", answer: "No. Every fixed-width integer is finite, so host and resident integer inputs can return true from exact class metadata." },
    BuiltinDocumentationFaq { question: "Does a gpuArray input return a gpuArray?", answer: "No. The full-array reduction returns one host logical scalar. Supported providers perform the classification and reduction before transferring that scalar." },
];

const RELATED: &[&str] = &["isfinite", "all", "isinf", "isnan", "gpuArray", "gather"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isfinite", target: BuiltinDocumentationLinkTarget::Builtin("isfinite") },
    BuiltinDocumentationLink { label: "all", target: BuiltinDocumentationLinkTarget::Builtin("all") },
    BuiltinDocumentationLink { label: "isinf", target: BuiltinDocumentationLinkTarget::Builtin("isinf") },
    BuiltinDocumentationLink { label: "isnan", target: BuiltinDocumentationLinkTarget::Builtin("isnan") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/allfinite.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Finite scalar-reduction runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/allfinite.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, complex, sparse, integer, text, empty, error, and provider behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/allfinite/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU reduction and host-scalar result", location: "crates/runmat-runtime/src/builtins/logical/tests/allfinite/tests.rs::wgpu_matches_host_and_returns_scalar" },
    ],
    notes: &[],
};

pub(super) const ALLFINITE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("allfinite"),
    slug: Some("allfinite"),
    summary: "Determine whether every element of an array is finite.",
    description: "`allfinite` reduces numeric, logical, character, and supported string input to one logical scalar.",
    keywords: &["allfinite", "finite", "isfinite", "all", "logical", "complex", "integer", "gpuArray"],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
