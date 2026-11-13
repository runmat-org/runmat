use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Shape and bounds",
        paragraphs: &[
            "All non-scalar subscript arrays must share their shape; scalars expand to that shape. The result is a double scalar or an array matching the common non-scalar shape. Every subscript must be a positive integer within its dimension bound.",
            "The number of subscript inputs must match the size-vector length. Invalid dimensions and noninteger, complex, nonfinite, or out-of-bounds subscripts produce typed errors.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Exact integer and resident behavior",
        paragraphs: &[
            "The size vector and each subscript accept all eight native integer classes, single, and double. Native integer inputs are checked from authoritative storage, including values above flintmax; the double result must remain exact.",
            "A provider hook handles supported resident floating-point forms. Native integer resident values take an exact owner-aware gather path; successful output is restored to the same owner when it can represent binary64. No fusion occurs.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "native-aot-contract",
        title: "Compile vector indices and bounds errors",
        program: "dims = [3 5]; idx = sub2ind(dims,[1;2;3],3); badrow = 4; caught = false; try; bad = sub2ind(dims,badrow,1); catch failure; caught = strcmp(failure.identifier,'RunMat:sub2ind:IndexBounds'); end",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            engine: BuiltinExampleEngine::Aot,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx,[7;8;9]) && isequal(size(idx),[3 1])); assert(isa(idx,'double') && caught);",
        },
    },
    BuiltinExample {
        id: "out-of-bounds",
        title: "Reject a subscript beyond its dimension",
        program: "idx = sub2ind([2 3], 3, 1)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "MATLAB:sub2ind:IndexBounds",
        },
    },
    BuiltinExample {
        id: "scalar",
        title: "Map a matrix coordinate",
        program: "idx = sub2ind([3 4], 2, 3)",
        display_output: Some("idx = 8"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(idx == 8);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Preserve vector orientation",
        program: "idx = sub2ind([3 5], [1;2;3], [3;3;3])",
        display_output: Some("idx = [7;8;9]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx, [7;8;9]));",
        },
    },
    BuiltinExample {
        id: "nd",
        title: "Map higher-dimensional coordinates",
        program: "idx = sub2ind([2 3 4], [1 1], [2 3], [1 2])",
        display_output: Some("idx = [3 11]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx, [3 11]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Are fractional subscripts rounded?",
        answer: "No. Fractional, nonfinite, and complex subscripts reject.",
    },
    BuiltinDocumentationFaq {
        question: "Do scalar subscripts broadcast?",
        answer: "Yes, to the common shape of the non-scalar subscripts.",
    },
];
const RELATED: &[&str] = &["ind2sub", "find", "size", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "ind2sub",
        target: BuiltinDocumentationLinkTarget::Builtin("ind2sub"),
    },
    BuiltinDocumentationLink {
        label: "find",
        target: BuiltinDocumentationLinkTarget::Builtin("find"),
    },
    BuiltinDocumentationLink {
        label: "size",
        target: BuiltinDocumentationLinkTarget::Builtin("size"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Indexing runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/indexing/sub2ind.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Indexing values, errors, integer storage, output modes, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/array/indexing/sub2ind.rs::tests",
    }],
    notes: &[],
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sub2ind"),
    slug: Some("sub2ind"),
    summary: "Convert N-D subscripts to one-based column-major linear indices.",
    description: "sub2ind combines one subscript per dimension of a positive integral size vector using checked column-major strides.",
    keywords: &["sub2ind", "indexing", "column major", "gpu"],
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
