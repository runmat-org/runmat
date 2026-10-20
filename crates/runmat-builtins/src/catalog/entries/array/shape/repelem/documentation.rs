use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("repelem"), slug: Some("repelem"), related: &[],
    summary: "Repeat each element of an array by per-dimension replication factors, matching MATLAB `repelem` semantics.",
    description: "`repelem(A, r1, r2, ..., rN)` repeats each element of `A` along the requested dimensions. Unlike `repmat`, which tiles whole arrays, `repelem` expands individual elements into repeated runs or blocks.",
    keywords: &["repelem", "replicate", "repeat", "array", "cell"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`repelem(v, n)` repeats each element of a row or column vector `v` `n` times and preserves vector orientation.",
            "`repelem(v, counts)` accepts a count vector with one non-negative integer count per element of `v`.",
            "`repelem(A, r1, r2, ..., rN)` repeats matrix or N-D array elements along each dimension; scalar factors create element blocks and vector factors provide per-index counts for that dimension.",
            "Numeric, logical, complex, string, char, and cell inputs are supported. Numeric, logical, complex, string, and cell arrays support N-D replication; char arrays are represented as 2-D character matrices.",
            "All eight integer data classes are replicated from authoritative native storage and returned in the same class; typed integer replication counts are decoded structurally without a floating round trip.",
            "Zero counts are allowed and remove the corresponding element or dimension slice from the output.",
            "Replication counts must be finite, non-negative integers. Count vectors must match the size of the dimension they apply to.",
            "The builtin checks shape and size overflow before materializing output arrays.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "GPU inputs and resident replication controls are gathered through their owning provider; the native-class host result is restored to the owning provider. No native replication hook is used.",
            "This structural operation is a fusion boundary, not an elementwise or reduction kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Repeating every element of a row vector", program: "u = repelem([1 2 3], 2)", display_output: Some("u =\n     1     1     2     2     3     3"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(u,[1 1 2 2 3 3]));" } },
        BuiltinExample { id: "example-2", title: "Using a per-element count vector", program: "u = repelem([1 2 3], [1 2 3])", display_output: Some("u =\n     1     2     2     3     3     3"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(u,[1 2 2 3 3 3]));" } },
        BuiltinExample { id: "example-3", title: "Expanding a matrix into element blocks", program: "A = [1 2; 3 4];\nB = repelem(A, 2, 3)", display_output: Some("B =\n     1     1     1     2     2     2\n     1     1     1     2     2     2\n     3     3     3     4     4     4\n     3     3     3     4     4     4"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B,[1 1 1 2 2 2;1 1 1 2 2 2;3 3 3 4 4 4;3 3 3 4 4 4]));" } },
        BuiltinExample { id: "example-4", title: "Dropping elements with zero counts", program: "u = repelem([1 2 3 4 5], [0 1 0 2 1])", display_output: Some("u =\n     2     4     4     5"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(u,[2 4 4 5]));" } },
        BuiltinExample { id: "example-5", title: "Repeating along a higher dimension", program: "C = {1; 2};\nD = repelem(reshape(C, [1 1 2]), 1, 1, 2);\nsize(D)", display_output: Some("ans =\n     1     1     4"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(D),[1 1 4]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is `repelem` different from `repmat`?", answer: "`repmat` tiles a whole array. `repelem` repeats each individual element, so matrix inputs expand into element-wise blocks." },
        BuiltinDocumentationFaq { question: "Can counts be different for each element?", answer: "Yes. In vector form, the count vector must have one entry for each vector element. In multi-dimensional form, a count vector applies to the dimension at the same argument position." },
        BuiltinDocumentationFaq { question: "Are zero counts valid?", answer: "Yes. A zero count omits that element or dimension index from the output." },
        BuiltinDocumentationFaq { question: "Does `repelem` preserve row and column vector orientation?", answer: "Yes. Row vectors remain row vectors, column vectors remain column vectors, including empty column vectors." },
        BuiltinDocumentationFaq { question: "Does `repelem` run on the GPU?", answer: "There is no native GPU replication hook. Resident data is gathered, replicated on the host, and restored with its original class and provider ownership." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "repmat", target: BuiltinDocumentationLinkTarget::Builtin("repmat") },
        BuiltinDocumentationLink { label: "kron", target: BuiltinDocumentationLinkTarget::Builtin("kron") },
        BuiltinDocumentationLink { label: "reshape", target: BuiltinDocumentationLinkTarget::Builtin("reshape") },
        BuiltinDocumentationLink { label: "cat", target: BuiltinDocumentationLinkTarget::Builtin("cat") },
        BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "repelem runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/repelem.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained repelem verification", location: "crates/runmat-runtime/tests/nd_ops.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Retained repelem verification", location: "builtins::array::shape::repelem::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "repelem semantic and provider tests", location: "crates/runmat-runtime/src/builtins/array/shape/repelem.rs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
