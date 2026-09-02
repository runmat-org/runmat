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
            "`isfinite(A)` returns a logical value for every element of `A`. Real values are true unless they are `NaN`, positive infinity, or negative infinity. A complex element is true only when both components are finite.",
            "All fixed-width integers and logical values are finite by construction. Character elements are finite Unicode code points. String scalars return false and string arrays produce same-shaped false masks. Sparse and unrelated container values are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Shape, residency, and fusion",
        paragraphs: &[
            "The output has the same shape as the input. Scalar input returns a logical scalar; nonscalar and empty input returns a logical array with the original dimensions.",
            "A resident floating input uses its exact provider's `logical_isfinite` operation when available. A typed unsupported result falls back through one class-preserving transfer and restores the logical mask to the same owner. Integer input can produce the constant mask from class and shape alone. Compatible elementwise expressions may fuse.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Classify a finite scalar",
        program: "tf = isfinite(42)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Build a finite-value mask",
        program: "A = [1 NaN; Inf 4];\ntf = isfinite(A)",
        display_output: Some("tf = [true false; false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0; 0 1])));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Require both complex components to be finite",
        program: "Z = complex([1 Inf NaN], [2 0 3]);\ntf = isfinite(Z)",
        display_output: Some("tf = [true false false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 0])));",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Classify character code points",
        program: "tf = isfinite(['R' 'u' 'n'])",
        display_output: Some("tf = [true true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 1 1])));",
        },
    },
    BuiltinExample {
        id: "integers",
        title: "Classify fixed-width integers exactly",
        program: "A = [uint64(0), uint64(9007199254740992) + uint64(1), intmax('uint64')];\ntf = isfinite(A)",
        display_output: Some("tf = [true true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 1 1])));",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Keep a logical mask resident",
        program: "A = gpuArray(single([1 -Inf NaN]));\ngtf = isfinite(A);\ntf = gather(gtf)",
        display_output: Some("gtf remains a gpuArray and tf = [true false false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(gtf, 'gpuArray'));\nassert(isequal(tf, logical([1 0 0])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Do NaN and infinity count as finite?", answer: "No. Use `isnan` and `isinf` when the two cases need to be distinguished." },
    BuiltinDocumentationFaq { question: "How are complex values classified?", answer: "An element is finite only when both its real and imaginary components are finite." },
    BuiltinDocumentationFaq { question: "How are logical, character, and string values classified?", answer: "Logical values and character code points are finite. String scalars return false, and string arrays produce same-shaped false masks." },
    BuiltinDocumentationFaq { question: "Are integer values converted to floating point?", answer: "No. Every fixed-width integer is finite, so RunMat builds the logical mask directly from the exact class and shape." },
    BuiltinDocumentationFaq { question: "Does a gpuArray input gather?", answer: "The owning provider is used when it implements the operation. A typed unsupported response uses a class-preserving fallback and restores residency." },
    BuiltinDocumentationFaq { question: "Can `isfinite` participate in a fused expression?", answer: "Yes. The placement and fusion planner can combine compatible elementwise work when the selected provider supports the complete region." },
];

const RELATED: &[&str] = &[
    "allfinite",
    "isinf",
    "isnan",
    "isreal",
    "gpuArray",
    "gather",
    "isgpuarray",
    "islogical",
    "isnumeric",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isinf", target: BuiltinDocumentationLinkTarget::Builtin("isinf") },
    BuiltinDocumentationLink { label: "isnan", target: BuiltinDocumentationLinkTarget::Builtin("isnan") },
    BuiltinDocumentationLink { label: "isreal", target: BuiltinDocumentationLinkTarget::Builtin("isreal") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "isgpuarray", target: BuiltinDocumentationLinkTarget::Builtin("isgpuarray") },
    BuiltinDocumentationLink { label: "islogical", target: BuiltinDocumentationLinkTarget::Builtin("islogical") },
    BuiltinDocumentationLink { label: "isnumeric", target: BuiltinDocumentationLinkTarget::Builtin("isnumeric") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isfinite.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Finite classification runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isfinite.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, complex, integer, text, empty, and error behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/isfinite/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU mask execution and residency", location: "crates/runmat-runtime/src/builtins/logical/tests/isfinite/tests.rs::wgpu_matches_host_and_preserves_residency" },
    ],
    notes: &[],
};

pub(super) const ISFINITE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isfinite"),
    slug: Some("isfinite"),
    summary: "Determine which array elements are finite.",
    description: "`isfinite` returns a same-shaped logical mask that distinguishes finite real or complex values from infinities and NaNs.",
    keywords: &["isfinite", "finite", "logical", "complex", "integer", "gpuArray", "classification"],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("Before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
