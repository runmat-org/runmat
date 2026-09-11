use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Storage class",
        paragraphs: &[
            "`islogical(A)` returns one logical scalar. It is true for logical scalars, logical arrays, and logical sparse matrices, regardless of shape or values. Numeric, character, string, container, object, callable, and symbolic values return false.",
            "The predicate does not convert its input. Use `logical(A)` when a numeric or sparse value should be converted to logical storage.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &["For a resident value, RunMat validates the exact owner's physical storage and typed class metadata, then returns a host logical scalar without downloading the payload. A logical value stays logical after `gather`."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Check a logical scalar", program: "tf = islogical(true)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" } },
    BuiltinExample { id: "array", title: "Check a logical array", program: "mask = logical([1 0 1 0]);\ntf = islogical(mask)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" } },
    BuiltinExample { id: "numeric", title: "Distinguish numeric from logical storage", program: "tf = islogical([1 2 3])", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, false));" } },
    BuiltinExample { id: "text", title: "Reject character and string storage", program: "tf_chars = islogical(['R' 'u' 'n']);\ntf_string = islogical(\"RunMat\")", display_output: Some("Both results are false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf_chars);\nassert(~tf_string);" } },
    BuiltinExample { id: "containers", title: "Reject container storage", program: "items = {true, 1};\nrecord = struct(\"enabled\", true);\ntf_cell = islogical(items);\ntf_struct = islogical(record)", display_output: Some("Both results are false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf_cell);\nassert(~tf_struct);" } },
    BuiltinExample { id: "gpu-logical", title: "Inspect a resident logical mask", program: "G = gpuArray([1 2 3]);\nmask = G > 1;\ntf = islogical(mask);\nhost_mask = gather(mask)", display_output: Some("tf = true and host_mask remains logical"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);\nassert(islogical(host_mask));\nassert(isequal(host_mask, logical([0 1 1])));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `islogical` return an array?",
        answer: "No. It returns one logical scalar describing the input's storage class.",
    },
    BuiltinDocumentationFaq {
        question: "Does it convert numeric values?",
        answer: "No. Use `logical` for explicit conversion.",
    },
    BuiltinDocumentationFaq {
        question: "Are characters, strings, cells, structs, tables, or objects logical?",
        answer: "No. Only values stored in the logical class return true.",
    },
    BuiltinDocumentationFaq {
        question: "Does a resident query gather the payload?",
        answer:
            "No. RunMat checks coherent owner and class metadata without downloading element data.",
    },
    BuiltinDocumentationFaq {
        question: "Does a gathered logical gpuArray stay logical?",
        answer: "Yes. Gathering preserves the logical class.",
    },
];

const RELATED: &[&str] = &[
    "logical",
    "isnumeric",
    "isreal",
    "issparse",
    "gpuArray",
    "gather",
    "isgpuarray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "logical", target: BuiltinDocumentationLinkTarget::Builtin("logical") },
    BuiltinDocumentationLink { label: "isnumeric", target: BuiltinDocumentationLinkTarget::Builtin("isnumeric") },
    BuiltinDocumentationLink { label: "isreal", target: BuiltinDocumentationLinkTarget::Builtin("isreal") },
    BuiltinDocumentationLink { label: "isgpuarray", target: BuiltinDocumentationLinkTarget::Builtin("isgpuarray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/islogical.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Logical-storage predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/islogical.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, sparse, resident, integer, and container behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/islogical/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU logical metadata", location: "crates/runmat-runtime/src/builtins/logical/tests/islogical/tests.rs::wgpu_metadata_matches_gathered_class" }],
    notes: &[],
};

pub(super) const ISLOGICAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("islogical"), slug: Some("islogical"), summary: "Determine whether a value uses logical storage.",
    description: "`islogical` returns one logical scalar that reports whether its input is stored in the logical class.",
    keywords: &["islogical", "logical", "boolean", "storage", "gpuArray"], related: RELATED,
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
