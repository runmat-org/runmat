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
            "`isinf(A)` returns true where an element is positive or negative infinity. A complex element is true when either component is infinite; a `NaN` component does not by itself count as infinite.",
            "Fixed-width integers, logical values, and character code points cannot be infinite and therefore produce false masks. String scalars return false and string arrays produce same-shaped false masks. Sparse and unrelated container values are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Shape, residency, and fusion",
        paragraphs: &[
            "The result is a logical scalar for scalar input and a same-shaped logical array for nonscalar or empty input.",
            "Resident floating input uses the exact owner's `logical_isinf` operation. A typed unsupported result may gather once and restore the logical output to that owner. Integer input needs no payload download because its class determines an all-false result. Compatible elementwise expressions may fuse.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Classify an infinite scalar",
        program: "tf = isinf(1 / 0)",
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
        title: "Find positive and negative infinity",
        program: "A = [1 Inf; -Inf NaN];\ntf = isinf(A)",
        display_output: Some("tf = [false true; true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1; 1 0])));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Classify either complex component",
        program: "Z = complex([1 Inf NaN], [Inf 0 2]);\ntf = isinf(Z)",
        display_output: Some("tf = [true true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 1 0])));",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Classify character code points",
        program: "tf = isinf(['R' 'u' 'n'])",
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
        program: "A = [intmin('int64'), int64(0), intmax('int64')];\ntf = isinf(A)",
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
        title: "Keep the infinity mask resident",
        program: "A = gpuArray(single([1 -Inf Inf]));\ngtf = isinf(A);\ntf = gather(gtf)",
        display_output: Some("gtf remains a gpuArray and tf = [false true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(gtf, 'gpuArray'));\nassert(isequal(tf, logical([0 1 1])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does NaN count as infinity?", answer: "No. Only positive or negative infinity is true. A complex value can independently contain an infinite component and a NaN component." },
    BuiltinDocumentationFaq { question: "How are logical, character, and string values classified?", answer: "Logical values, character code points, and strings cannot contain infinity, so they produce false values with the corresponding shape." },
    BuiltinDocumentationFaq { question: "Are integers converted?", answer: "No. An integer class cannot contain infinity, so RunMat constructs the false mask directly." },
    BuiltinDocumentationFaq { question: "Does a gpuArray result remain resident?", answer: "Yes when the exact owner can execute the operation or restore the fallback result without losing explicit residency." },
    BuiltinDocumentationFaq { question: "Can `isinf` participate in a fused expression?", answer: "Yes. The placement and fusion planner can combine compatible elementwise work when the selected provider supports the complete region." },
];

const RELATED: &[&str] = &[
    "isfinite",
    "isnan",
    "isreal",
    "gpuArray",
    "gather",
    "isgpuarray",
    "islogical",
    "isnumeric",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isfinite", target: BuiltinDocumentationLinkTarget::Builtin("isfinite") },
    BuiltinDocumentationLink { label: "isnan", target: BuiltinDocumentationLinkTarget::Builtin("isnan") },
    BuiltinDocumentationLink { label: "isreal", target: BuiltinDocumentationLinkTarget::Builtin("isreal") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "isgpuarray", target: BuiltinDocumentationLinkTarget::Builtin("isgpuarray") },
    BuiltinDocumentationLink { label: "islogical", target: BuiltinDocumentationLinkTarget::Builtin("islogical") },
    BuiltinDocumentationLink { label: "isnumeric", target: BuiltinDocumentationLinkTarget::Builtin("isnumeric") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isinf.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Infinity classification runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isinf.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, complex, integer, text, empty, and error behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/isinf/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU mask execution and residency", location: "crates/runmat-runtime/src/builtins/logical/tests/isinf/tests.rs::wgpu_matches_host_and_preserves_residency" },
    ],
    notes: &[],
};

pub(super) const ISINF_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isinf"),
    slug: Some("isinf"),
    summary: "Determine which array elements are infinite.",
    description: "`isinf` returns a same-shaped logical mask that identifies positive or negative infinity in real and complex values.",
    keywords: &["isinf", "infinity", "logical", "complex", "integer", "gpuArray", "classification"],
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
