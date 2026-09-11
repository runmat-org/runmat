use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Element count", paragraphs: &["`numel(A)` returns the product of the MATLAB-visible dimensions of `A` as a host double scalar. It counts outer array elements: cells rather than their contents, table rows times variables, string elements rather than characters within each string, and every character in a character array.", "Any zero extent makes the result zero. Checked structural arithmetic detects an overflow before a result is formed, and RunMat rejects a count that the documented double result cannot represent exactly."] },
    BuiltinDocumentationSection { heading: "Selected-dimension extension", paragraphs: &["In RunMat compatibility mode, `numel(A, dim1, dim2, ...)` returns the checked product of selected extents. One dimension vector is also accepted. All eight integer classes are decoded exactly, dimensions beyond the visible rank contribute 1, and an empty selector is rejected.", "The selected-dimension forms are a named RunMat extension. A MATLAB compatibility policy accepts only `numel(A)`, matching the public compatible syntax."] },
    BuiltinDocumentationSection { heading: "Resident and distributed values", paragraphs: &["gpuArray and distributed inputs are counted from validated shape metadata. `numel` launches no kernel, calls no acceleration provider, transfers no payload, and returns one host scalar at a fusion boundary."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix",
        title: "Count matrix elements",
        program: "A = [1 2 3; 4 5 6];\nn = numel(A)",
        display_output: Some("n = 6"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 6);",
        },
    },
    BuiltinExample {
        id: "n-dimensional",
        title: "Count an N-D array",
        program: "A = zeros(4, 4, 2);\nn = numel(A)",
        display_output: Some("n = 32"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 32);",
        },
    },
    BuiltinExample {
        id: "cell",
        title: "Count cells without inspecting their contents",
        program: "C = {1, 2, 3; 4, 5, 6};\nn = numel(C)",
        display_output: Some("n = 6"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 6);",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Count characters in a character row",
        program: "name = 'RunMat';\nn = numel(name)",
        display_output: Some("n = 6"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 6);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "Count an empty array",
        program: "A = zeros(0, 7);\nn = numel(A)",
        display_output: Some("n = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 0);",
        },
    },
    BuiltinExample {
        id: "selected-dimensions",
        title: "Multiply selected dimensions",
        program: "A = zeros(4, 3, 2);\nn = numel(A, 1, 2)",
        display_output: Some("n = 12"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 12);",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Count a resident array without gathering",
        program: "G = gpuArray(ones(256, 4));\nn = numel(G)",
        display_output: Some("n = 1024"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 1024);\nassert(~isgpuarray(n));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `numel` different from `length`?", answer: "`numel` multiplies every dimension; `length` returns only the largest dimension." },
    BuiltinDocumentationFaq { question: "When should I use `size` instead?", answer: "Use `size` when individual extents or the complete shape are needed." },
    BuiltinDocumentationFaq { question: "What does `numel` count in a cell array or struct array?", answer: "It counts outer cells or struct elements and does not recurse into their contents or fields." },
    BuiltinDocumentationFaq { question: "How are strings and character arrays different?", answer: "A string scalar is one string-array element. A character row has one element per character." },
    BuiltinDocumentationFaq { question: "What happens for an empty array?", answer: "The product is zero when any visible extent is zero." },
    BuiltinDocumentationFaq { question: "Are selected dimensions compatible syntax?", answer: "No. They are enabled only by RunMat compatibility mode; the compatible public form is `numel(A)`." },
    BuiltinDocumentationFaq { question: "Does `numel` gather gpuArray or distributed data?", answer: "No. Complete shape metadata is part of each validated handle." },
];

const RELATED: &[&str] = &["size", "length", "ndims", "isempty", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/numel.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shared shape-query primitives", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/array/introspection/shape_query") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Array, container, text, table, resident, compatibility, selector, exactness, and output-count semantics", location: "crates/runmat-runtime/src/builtins/array/introspection/numel.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed scalar-result inference", location: "crates/runmat-builtins/src/catalog/inference/array/introspection/shape_query.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Assertion-backed resident metadata example", location: "numel::gpu-shape" },
    ], notes: &[],
};

pub(super) const NUMEL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("numel"), slug: Some("numel"),
    summary: "Return the number of outer array elements.",
    description: "`numel` computes an exact checked product of visible dimensions and includes a compatibility-gated RunMat extension for selected dimensions.",
    keywords: &["numel", "number of elements", "element count", "array size", "gpu metadata", "distributed"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};
