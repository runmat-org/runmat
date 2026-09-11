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
            "`isnan(A)` returns true where an element contains an IEEE NaN. A complex element is true when either its real or imaginary component is NaN.",
            "Fixed-width integers, logical values, and character code points cannot contain NaN and therefore produce false masks. String scalars return false and string arrays produce same-shaped false masks. Sparse and unrelated container values are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Shape, residency, and fusion",
        paragraphs: &[
            "The result is a logical scalar for scalar input and a same-shaped logical array for nonscalar or empty input.",
            "Resident floating input uses the exact owner's `logical_isnan` operation. A typed unsupported result may gather once and restore the logical output to that owner. Integer input needs no payload download because its class determines an all-false result. Compatible elementwise expressions may fuse.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Classify a NaN scalar",
        program: "tf = isnan(NaN)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Build a NaN mask",
        program: "A = [1 NaN 2; 3 4 NaN];\ntf = isnan(A)",
        display_output: Some("tf = [false true false; false false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 0; 0 0 1])));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Classify either complex component",
        program: "Z = complex([1 NaN 3], [2 0 NaN]);\ntf = isnan(Z)",
        display_output: Some("tf = [false true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 1])));",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Classify character code points",
        program: "tf = isnan(['R' 'u' 'n'])",
        display_output: Some("tf = [false false false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 0 0])));",
        },
    },
    BuiltinExample {
        id: "integers",
        title: "Classify fixed-width integers exactly",
        program: "A = [uint64(0), uint64(9007199254740992) + uint64(1), intmax('uint64')];\ntf = isnan(A)",
        display_output: Some("tf = [false false false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 0 0])));",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Keep the NaN mask resident",
        program: "A = gpuArray(single([1 NaN 3]));\ngtf = isnan(A);\ntf = gather(gtf)",
        display_output: Some("gtf remains a gpuArray and tf = [false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(gtf, 'gpuArray'));\nassert(isequal(tf, logical([0 1 0])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `isnan` modify its input?", answer: "No. It returns a new logical mask and leaves host- or provider-resident input unchanged." },
    BuiltinDocumentationFaq { question: "How are complex values classified?", answer: "An element is true when either its real or imaginary component is NaN. Infinity alone does not count as NaN." },
    BuiltinDocumentationFaq { question: "How are logical, character, and string values classified?", answer: "These classes cannot contain NaN, so they produce false values with the corresponding shape." },
    BuiltinDocumentationFaq { question: "How should NaN values be filtered?", answer: "Use the logical result as an index or combine it with `any` along the dimension whose rows or columns should be selected." },
    BuiltinDocumentationFaq { question: "Does gpuArray input use a provider operation?", answer: "Yes. RunMat asks the exact owner for `logical_isnan`; only a typed unsupported result enters the class-preserving fallback path." },
    BuiltinDocumentationFaq { question: "What happens for an empty array?", answer: "The result is an empty logical array with the same dimensions as the input." },
    BuiltinDocumentationFaq { question: "How do `isnan`, `isinf`, and `isfinite` differ?", answer: "`isnan` identifies NaN, `isinf` identifies positive or negative infinity, and `isfinite` identifies values that are neither. For an ordinary real floating value, exactly one predicate is true." },
    BuiltinDocumentationFaq { question: "Can `isnan` participate in a fused expression?", answer: "Yes. The placement and fusion planner can combine compatible elementwise work when the selected provider supports the complete region." },
];

const RELATED: &[&str] = &[
    "isfinite",
    "isinf",
    "isreal",
    "gpuArray",
    "gather",
    "isgpuarray",
    "islogical",
    "isnumeric",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isfinite", target: BuiltinDocumentationLinkTarget::Builtin("isfinite") },
    BuiltinDocumentationLink { label: "isinf", target: BuiltinDocumentationLinkTarget::Builtin("isinf") },
    BuiltinDocumentationLink { label: "isreal", target: BuiltinDocumentationLinkTarget::Builtin("isreal") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "isgpuarray", target: BuiltinDocumentationLinkTarget::Builtin("isgpuarray") },
    BuiltinDocumentationLink { label: "islogical", target: BuiltinDocumentationLinkTarget::Builtin("islogical") },
    BuiltinDocumentationLink { label: "isnumeric", target: BuiltinDocumentationLinkTarget::Builtin("isnumeric") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isnan.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "NaN classification runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isnan.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, complex, integer, text, empty, and error behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/isnan/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU mask execution and residency", location: "crates/runmat-runtime/src/builtins/logical/tests/isnan/tests.rs::wgpu_matches_host_and_preserves_residency" },
    ],
    notes: &[],
};

pub(super) const ISNAN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isnan"),
    slug: Some("isnan"),
    summary: "Determine which array elements are NaN.",
    description: "`isnan` returns a same-shaped logical mask that identifies NaN components in real and complex values.",
    keywords: &["isnan", "NaN", "logical", "complex", "integer", "gpuArray", "classification"],
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
