use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Dimensions and requested outputs",
        paragraphs: &[
            "The size vector contains at least two positive integral extents. Each requested output is a double subscript array. Fewer requested outputs fold trailing dimensions into the final output; with one output this is the original linear index. Extra outputs extend singleton dimensions.",
            "Each output array follows the input index shape, including empty and N-D forms. Positive indices beyond the nominal product extend the final effective dimension, subject to exact structural bounds. An internal callable without a requested-output count retains a cell carrier for all per-dimension outputs.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Numeric and resident behavior",
        paragraphs: &[
            "Size extents and indices accept single, double, and all eight native integer classes; wide integers are decoded exactly. Subscripts are doubles. Complex, fractional, nonfinite, and nonpositive values reject.",
            "The binary64 WGPU route can execute admitted forms on device. Other providers use exact owner-aware materialization and restoration where binary64 output is representable. No fusion occurs.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "native-aot-contract",
        title: "Compile requested subscripts and errors",
        program: "sz = [3 4]; indices = [7 8 9]; [row,col] = ind2sub(sz,indices); linear = ind2sub(sz,indices); invalid = [2 0]; caught = false; try; bad = ind2sub(invalid,1); catch failure; caught = strcmp(failure.identifier,'RunMat:ind2sub:InvalidInput'); end",
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
            source: "assert(isequal(row,[1 2 3]) && isequal(col,[3 3 3])); assert(isequal(linear,indices)); assert(isequal(size(row),[1 3]) && isequal(size(col),[1 3])); assert(isa(row,'double') && isa(col,'double') && caught);",
        },
    },
    BuiltinExample {
        id: "invalid-size",
        title: "Reject a zero dimension",
        program: "I = ind2sub([2 0], 1)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "MATLAB:ind2sub:InvalidInput",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Recover row and column",
        program: "[row,col] = ind2sub([3 4], 8)",
        display_output: Some("row = 2, col = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(row == 2 && col == 3);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Convert several indices",
        program: "[row,col] = ind2sub([3 5], [7 8 9])",
        display_output: Some("row = [1 2 3], col = [3 3 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(row, [1 2 3]));\nassert(isequal(col, [3 3 3]));",
        },
    },
    BuiltinExample {
        id: "single-output",
        title: "Collapse dimensions into one output",
        program: "I = ind2sub([3 4], [1 8])",
        display_output: Some("I = [1 8]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(I, [1 8]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What happens with fewer outputs?",
        answer: "Trailing dimensions collapse into the final requested subscript.",
    },
    BuiltinDocumentationFaq {
        question: "Do output shapes match the input indices?",
        answer: "Yes, including empty arrays and N-D index arrays.",
    },
];
const RELATED: &[&str] = &["sub2ind", "size", "find", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sub2ind",
        target: BuiltinDocumentationLinkTarget::Builtin("sub2ind"),
    },
    BuiltinDocumentationLink {
        label: "size",
        target: BuiltinDocumentationLinkTarget::Builtin("size"),
    },
    BuiltinDocumentationLink {
        label: "find",
        target: BuiltinDocumentationLinkTarget::Builtin("find"),
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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/indexing/ind2sub.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Indexing values, errors, integer storage, output modes, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/array/indexing/ind2sub.rs::tests",
    }],
    notes: &[],
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ind2sub"),
    slug: Some("ind2sub"),
    summary: "Convert one-based linear indices to N-D subscripts.",
    description: "ind2sub converts column-major linear indices using a positive integral size vector. Requested output count controls trailing-dimension collapse; an internal call with no requested-output context uses a cell carrier.",
    keywords: &["ind2sub", "indexing", "column major", "gpu"],
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
