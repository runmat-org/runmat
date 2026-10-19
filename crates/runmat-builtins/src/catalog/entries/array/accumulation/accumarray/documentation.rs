use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Indices and output shape", paragraphs: &["Each row of `ind` selects one output element. A vector creates a column result, a matrix supplies one subscript per column, and a cell array of vectors supplies the same columns separately.", "`sz` can provide the output dimensions explicitly. With `[]` or an omitted size, the largest subscript in each dimension determines the result shape."] },
    BuiltinDocumentationSection { heading: "Group computation and fill", paragraphs: &["The default computation sums each group and returns double. A supplied function receives one column vector per occupied output element and must return a supported scalar; its result class determines the full output class.", "Unoccupied elements use `fillval`. The fill must have the callback result class. With no explicit fill, numeric and logical results use the matching zero value; cell-valued results require an explicit fill."] },
    BuiltinDocumentationSection { heading: "Sparse and resident inputs", paragraphs: &["Sparse output is limited to two dimensions and requires double data, double scalar callback results, and an omitted or double-zero fill.", "Provider-resident inputs are materialized by their owner before accumulation. The compatible GPU-array form accepts logical, single, and double data; resident fixed-width integer data and fills are rejected before transfer."] },
    BuiltinDocumentationSection { heading: "Exact integer controls", paragraphs: &["All fixed-width integer classes are accepted for positive indices and output dimensions and are decoded without conversion through double. Integer data reaches an explicit callback in its native class. The default sum of integer data returns double."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "sum",
        title: "Sum values by output index",
        program: "B = accumarray([1; 1; 2], [10; 20; 5])",
        display_output: Some("B = [30; 5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B, [30; 5]));",
        },
    },
    BuiltinExample {
        id: "matrix-indices",
        title: "Accumulate into a matrix",
        program: "ind = [1 1; 2 1; 1 2];\nB = accumarray(ind, [4; 5; 6])",
        display_output: Some("B = [4 6; 5 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B, [4 6; 5 0]));",
        },
    },
    BuiltinExample {
        id: "callback-fill",
        title: "Apply a function and fill empty groups",
        program: "B = accumarray([1; 1; 3], [4; 2; 9], [3 1], @min, 0)",
        display_output: Some("B = [2; 0; 9]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B, [2; 0; 9]));",
        },
    },
    BuiltinExample {
        id: "integer-callback",
        title: "Preserve integer data through a callback",
        program: "B = accumarray(uint8([1; 1; 2]), int16([7; 3; 9]), uint8([3 1]), @min, int16(0))",
        display_output: Some("B = int16([3; 9; 0])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(B, \"int16\"));\nassert(isequal(B, int16([3; 9; 0])));",
        },
    },
    BuiltinExample {
        id: "sparse",
        title: "Create a sparse result",
        program: "B = accumarray([1; 3], [2; 4], [4 1], [], [], true)",
        display_output: Some("A sparse 4-by-1 result with two stored values"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(B));\nassert(isequal(full(B), [2; 0; 4; 0]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How are multidimensional indices represented?", answer: "Each row of a numeric index matrix is one output subscript tuple. A cell array can provide the same tuple components as separate equal-length vectors." },
    BuiltinDocumentationFaq { question: "What class does accumarray return?", answer: "Default summation returns double. With an explicit group function, the function's supported scalar result class determines the output class and the fill must match it." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "groupcounts",
        target: BuiltinDocumentationLinkTarget::Builtin("groupcounts"),
    },
    BuiltinDocumentationLink {
        label: "sparse",
        target: BuiltinDocumentationLinkTarget::Builtin("sparse"),
    },
    BuiltinDocumentationLink {
        label: "Compatible accumarray reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/accumarray.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Indexed accumulation runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/accumulation/accumarray/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Indices, shapes, callbacks, typed integers, sparse output, providers, and errors", location: "crates/runmat-runtime/src/builtins/array/accumulation/accumarray/tests" }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("accumarray"), slug: Some("accumarray"), summary: "Accumulate values into array elements selected by one-based indices.", description: "`accumarray` groups scalar or vector data by one- or multidimensional output indices, applies a group computation, and materializes a full or sparse result.", keywords: &["accumarray", "accumulate", "indices", "groups", "sparse", "integer"], related: &["groupcounts", "splitapply", "sparse"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };
