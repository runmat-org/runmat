use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Selection and output shape",
        paragraphs: &[
            "The optional positive integer K limits the matches. 'first' scans from the start; 'last' selects the final K matches but reports them in ascending order. A row-vector input gives row-oriented one-output indices; other inputs give a column. Two or three outputs are column vectors, with trailing N-D dimensions folded into the column subscript.",
            "A zero scalar or empty matrix produces the empty matrix convention. Other empty row vectors retain row orientation. NaN and infinity count as nonzero; either nonzero complex component counts.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Compatibility and exactness",
        paragraphs: &[
            "find(X,direction) is a RunMat-only convenience form; compatible direction selection supplies K. Typed-integer sparse values are likewise gated. All eight integer classes use authoritative storage for zero testing and the third value output.",
            "Indices are doubles and reject values outside their exact range. The third output retains logical, single, complex, or integer storage as appropriate.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &[
            "A capable binary64 provider can run find directly. Other resident inputs use owner-aware exact gathering and return each output to the same owner when the class can be represented. find is eager and is not fused.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "native-aot-contract",
        title: "Compile selection outputs and errors",
        program: "A = [0 4 0; 7 0 9]; k = find(A); [r,c,v] = find(A); tail = find([1 0 2 3 0],2,'last'); direction = 'sideways'; caught = false; try; bad = find(A,1,direction); catch failure; caught = strcmp(failure.identifier,'RunMat:find:InvalidInput'); end",
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
            source: "assert(isequal(k,[2;3;6]) && isequal(size(k),[3 1]) && isa(k,'double')); assert(isequal(r,[2;1;2]) && isequal(c,[1;2;3]) && isequal(v,[7;4;9])); assert(isequal(size(r),[3 1]) && isequal(size(c),[3 1]) && isequal(size(v),[3 1])); assert(isequal(tail,[3 4]) && caught);",
        },
    },
    BuiltinExample {
        id: "invalid-direction",
        title: "Reject an unknown direction",
        program: "find([1 0], 1, 'sideways')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "MATLAB:find:InvalidInput",
        },
    },
    BuiltinExample {
        id: "linear",
        title: "Find matrix positions",
        program: "A = [0 4 0; 7 0 9];\nk = find(A)",
        display_output: Some("k = [2; 3; 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(k, [2; 3; 6]));",
        },
    },
    BuiltinExample {
        id: "last",
        title: "Select the final matches",
        program: "A = [1 0 2 3 0];\nk = find(A,2,'last')",
        display_output: Some("k = [3 4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(k, [3 4]));",
        },
    },
    BuiltinExample {
        id: "integer-values",
        title: "Preserve wide integer values",
        program: "A = uint64([0 0x0020000000000001u64]);\n[r,c,v] = find(A)",
        display_output: Some("r = 1, c = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(r == 1 && c == 2);\nassert(isa(v, \"uint64\"));\nassert(v == uint64(0x0020000000000001u64));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What counts as nonzero?",
        answer: "Exact nonzero values, NaN, and infinities; either component of a complex value can make it nonzero.",
    },
    BuiltinDocumentationFaq {
        question: "Does the value output preserve integer precision?",
        answer: "Yes. It reads and returns authoritative integer storage without a double round trip.",
    },
];
const RELATED: &[&str] = &["ind2sub", "sub2ind", "logical", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "ind2sub",
        target: BuiltinDocumentationLinkTarget::Builtin("ind2sub"),
    },
    BuiltinDocumentationLink {
        label: "sub2ind",
        target: BuiltinDocumentationLinkTarget::Builtin("sub2ind"),
    },
    BuiltinDocumentationLink {
        label: "logical",
        target: BuiltinDocumentationLinkTarget::Builtin("logical"),
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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/indexing/find.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Indexing values, errors, integer storage, output modes, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/array/indexing/find.rs::tests",
    }],
    notes: &[],
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("find"),
    slug: Some("find"),
    summary: "Locate nonzero indices and selected values in column-major order.",
    description: "find returns one-based double indices of nonzero elements. With three outputs it also returns selected values in their original class, including exact wide integers.",
    keywords: &["find", "indexing", "column major", "gpu"],
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
