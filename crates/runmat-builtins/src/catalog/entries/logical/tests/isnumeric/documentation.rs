use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Numeric classes", paragraphs: &["`isnumeric(A)` is true for real or complex `double`, `single`, and all eight fixed-width integer classes. It applies to scalars, dense arrays, and numeric sparse matrices. Logical sparse and dense values are not numeric.", "Character, string, symbolic, container, object, callable, and execution values return false. The result is always one logical scalar."] },
    BuiltinDocumentationSection { heading: "Resident values", paragraphs: &["RunMat validates a resident value against its exact provider owner and typed physical-storage metadata. Numeric and complex handles return true; logical handles return false. The query does not download the payload."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Check a numeric scalar", program: "tf = isnumeric(42)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "matrix", title: "Check a numeric matrix", program: "A = [1 2 3; 4 5 6];\ntf = isnumeric(A)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "complex", title: "Recognize complex numeric storage", program: "z = 1 + 2i;\ntf = isnumeric(z)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "integers", title: "Recognize exact fixed-width integers", program: "ids = [uint64(9007199254740992) + uint64(1), intmax('uint64')];\ntf = isnumeric(ids)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);\nassert(isa(ids, 'uint64'));" } },
    BuiltinExample { id: "logical-and-text", title: "Distinguish nonnumeric classes", program: "tf_logical = isnumeric(logical([1 0]));\ntf_chars = isnumeric(['R' 'M']);\ntf_string = isnumeric(\"RunMat\")", display_output: Some("All results are false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf_logical);\nassert(~tf_chars);\nassert(~tf_string);" } },
    BuiltinExample { id: "gpu-classes", title: "Inspect resident numeric and logical classes", program: "G = gpuArray(single([1 2 3]));\nmask = G > 1;\ntf_numeric = isnumeric(G);\ntf_mask = isnumeric(mask)", display_output: Some("tf_numeric = true and tf_mask = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(tf_numeric);\nassert(~tf_mask);" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `isnumeric` return an array?",
        answer: "No. It returns one logical scalar.",
    },
    BuiltinDocumentationFaq {
        question: "Are complex and sparse numeric arrays numeric?",
        answer: "Yes. Real, complex, dense, and numeric sparse storage return true.",
    },
    BuiltinDocumentationFaq {
        question: "Are logical values numeric?",
        answer: "No. Use `islogical` to detect logical storage.",
    },
    BuiltinDocumentationFaq {
        question: "Are characters or strings numeric?",
        answer: "No. Text classes return false.",
    },
    BuiltinDocumentationFaq {
        question: "Are cells, structs, tables, or objects numeric?",
        answer: "No. Container and object classes return false.",
    },
    BuiltinDocumentationFaq {
        question: "Does the resident query gather data?",
        answer: "No. It uses validated owner and class metadata.",
    },
];
const RELATED: &[&str] = &[
    "islogical",
    "isreal",
    "issparse",
    "isa",
    "gpuArray",
    "gather",
    "isgpuarray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "islogical", target: BuiltinDocumentationLinkTarget::Builtin("islogical") },
    BuiltinDocumentationLink { label: "isreal", target: BuiltinDocumentationLinkTarget::Builtin("isreal") },
    BuiltinDocumentationLink { label: "issparse", target: BuiltinDocumentationLinkTarget::Builtin("issparse") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isnumeric.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Numeric-storage predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isnumeric.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, dense, sparse, integer, logical, container, and resident behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/isnumeric/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU numeric and logical metadata", location: "crates/runmat-runtime/src/builtins/logical/tests/isnumeric/tests.rs::wgpu_metadata_matches_gathered_class" }], notes: &[] };
pub(super) const ISNUMERIC_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("isnumeric"), slug: Some("isnumeric"), summary: "Determine whether a value uses a numeric class.", description: "`isnumeric` returns one logical scalar for real or complex floating-point and fixed-width integer storage.", keywords: &["isnumeric", "numeric", "integer", "complex", "sparse", "gpuArray"], related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };
