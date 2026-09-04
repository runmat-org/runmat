use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Permutation order and shape",
        paragraphs: &[
            "`perms(v)` returns every positional permutation of scalar or vector `v`, one permutation per row. The rows follow reverse lexicographic order relative to the original element positions.",
            "The output has `factorial(numel(v))` rows and `numel(v)` columns. Duplicate values are still permuted by position, so repeated output rows are retained. An empty vector produces one empty row.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and placement",
        paragraphs: &[
            "Dense numeric, complex, logical, character, string-array, and cell-array vectors retain their container and element representation. Fixed-width integer values remain in native storage, including exact 64-bit values.",
            "Provider-resident input is gathered through its owning provider, permuted on the host, and restored with the same class and placement intent. Explicit device input returns an error when its owner cannot restore that contract.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "numeric-vector",
        title: "Permute a numeric vector",
        program: "P = perms([1 2 3])",
        display_output: Some("P = [3 2 1; 3 1 2; 2 3 1; 2 1 3; 1 3 2; 1 2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(P, [3 2 1; 3 1 2; 2 3 1; 2 1 3; 1 3 2; 1 2 3]));" },
    },
    BuiltinExample {
        id: "characters",
        title: "Permute characters",
        program: "P = perms('abc')",
        display_output: Some("P contains cba, cab, bca, bac, acb, and abc"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(P, ['cba'; 'cab'; 'bca'; 'bac'; 'acb'; 'abc']));" },
    },
    BuiltinExample {
        id: "duplicates",
        title: "Retain duplicate positional permutations",
        program: "P = perms([1 1 2])",
        display_output: Some("P has six rows, including duplicate rows"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(P), [6 3]));\nassert(sum(all(P == [2 1 1], 2)) == 2);" },
    },
    BuiltinExample {
        id: "empty",
        title: "Permute an empty vector",
        program: "P = perms([])",
        display_output: Some("P is a 1-by-0 empty double array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(P), [1 0]));" },
    },
    BuiltinExample {
        id: "wide-integer",
        title: "Preserve wide integer values",
        program: "v = uint64([0x0020000000000001u64 0xFFFFFFFFFFFFFFFFu64]);\nP = perms(v)",
        display_output: Some("P is a 2-by-2 uint64 array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(P, \"uint64\"));\nassert(isequal(size(P), [2 2]));\nassert(P(1,1) == v(2) && P(2,1) == v(1));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does perms remove duplicate rows?", answer: "No. It permutes positions, so repeated input values can produce repeated rows." },
    BuiltinDocumentationFaq { question: "Why can perms reject a long vector?", answer: "The result contains `factorial(numel(v)) * numel(v)` elements. RunMat checks this size before allocating the output." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "nchoosek",
        target: BuiltinDocumentationLinkTarget::Builtin("nchoosek"),
    },
    BuiltinDocumentationLink {
        label: "randperm",
        target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
    },
    BuiltinDocumentationLink {
        label: "Compatible perms reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/perms.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Deterministic permutations runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/combinatorics/perms/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Ordering, classes, containers, cardinality, errors, and provider restoration", location: "crates/runmat-runtime/src/builtins/array/combinatorics/perms/tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Integer transform execution", location: "crates/runmat-runtime/tests/integer_numeric_transform_semantics.rs" },
    ],
    notes: &[],
};

pub(super) const PERMS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("perms"),
    slug: Some("perms"),
    summary: "Enumerate every positional permutation of a vector.",
    description: "`perms` materializes reverse-lexicographic positional permutations while preserving supported container and numeric classes.",
    keywords: &["perms", "permutation", "combinatorics", "vector"],
    related: &["nchoosek", "randperm", "factorial"],
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
