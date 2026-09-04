use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Coefficient and vector forms",
        paragraphs: &[
            "`nchoosek(n,k)` returns the binomial coefficient for selecting `k` items from the nonnegative integer scalar `n`. `nchoosek(v,k)` returns every positional `k`-element combination of vector `v`, one combination per row.",
            "The vector form preserves supported numeric, complex, logical, and character data without converting element values. Its output has `k` columns. `k = 0` produces one empty row; `k > numel(v)` produces zero rows.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Size and execution",
        paragraphs: &["Combination counts grow quickly. RunMat validates the output cardinality before allocation and returns a stable error when the materialized result would exceed the supported limit. This builtin executes on the host; provider-resident inputs are not accepted."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "coefficient",
        title: "Compute a binomial coefficient",
        program: "b = nchoosek(5, 2)",
        display_output: Some("b = 10"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(b, 10));",
        },
    },
    BuiltinExample {
        id: "numeric-vector",
        title: "Enumerate numeric combinations",
        program: "C = nchoosek([10 20 30], 2)",
        display_output: Some("C = [10 20; 10 30; 20 30]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C, [10 20; 10 30; 20 30]));",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Enumerate character combinations",
        program: "C = nchoosek('abc', 2)",
        display_output: Some("C = ['ab'; 'ac'; 'bc']"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C, ['ab'; 'ac'; 'bc']));",
        },
    },
    BuiltinExample {
        id: "empty-selection",
        title: "Select zero elements",
        program: "C = nchoosek([1 2 3], 0)",
        display_output: Some("C is a 1-by-0 double array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C), [1 0]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why can nchoosek(v,k) reject a large vector?", answer: "The output has one row for every combination. RunMat checks the full element count before allocating the matrix." },
    BuiltinDocumentationFaq { question: "Does nchoosek execute on a GPU?", answer: "No. `nchoosek` is a host combinatorial operation and rejects provider-resident inputs." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "perms",
        target: BuiltinDocumentationLinkTarget::Builtin("perms"),
    },
    BuiltinDocumentationLink {
        label: "factorial",
        target: BuiltinDocumentationLinkTarget::Builtin("factorial"),
    },
    BuiltinDocumentationLink {
        label: "Compatible nchoosek reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/nchoosek.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Deterministic combinations runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/combinatorics/nchoosek/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coefficient, class, shape, container, cardinality, and error behavior", location: "crates/runmat-runtime/src/builtins/array/combinatorics/nchoosek/tests" }],
    notes: &[],
};

pub(super) const NCHOOSEK_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nchoosek"),
    slug: Some("nchoosek"),
    summary: "Compute a binomial coefficient or enumerate vector combinations.",
    description: "`nchoosek` computes a scalar selection count or materializes positional combinations while preserving the vector element class.",
    keywords: &["nchoosek", "combinations", "binomial", "combinatorics", "vector"],
    related: &["perms", "factorial"],
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
