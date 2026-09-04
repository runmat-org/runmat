use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Cartesian product and ordering",
        paragraphs: &[
            "`combinations(A1,...,An)` returns one table variable per input and one row per element tuple in the Cartesian product. Inputs are linearized in column-major order before their values are combined.",
            "The rightmost input changes fastest. Output variable names are `Var1`, `Var2`, and so on; text such as `VariableNames` is input data rather than an option name.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes, empty inputs, and placement",
        paragraphs: &[
            "Dense real numeric and logical classes are preserved in their corresponding table variables. A character row contributes string values, string arrays remain strings, and cell inputs retain their contained values. Other containers are retained as values in a cell variable. Fixed-width integers are repeated from native storage without conversion through `double`.",
            "If any input is empty, the table has zero rows and every variable remains present with its corresponding output class. The table is host-resident. In RunMat compatibility mode, provider-resident inputs may be gathered before the table is assembled.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "numeric-and-text",
        title: "Generate numeric and text pairs",
        program: "T = combinations([1 2], [\"x\" \"y\"])",
        display_output: Some("T is a 4-by-2 table containing every pair"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(height(T) == 4);\nassert(width(T) == 2);\nassert(isequal(T.Var1, [1; 1; 2; 2]));\nassert(isequal(T.Var2, [\"x\"; \"y\"; \"x\"; \"y\"]));" },
    },
    BuiltinExample {
        id: "three-inputs",
        title: "Build a three-input Cartesian product",
        program: "T = combinations([10 20], true, [3 4 5])",
        display_output: Some("T is a 6-by-3 table"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(T), [6 3]));\nassert(isequal(T.Var1, [10; 10; 10; 20; 20; 20]));\nassert(all(T.Var2));\nassert(isequal(T.Var3, [3; 4; 5; 3; 4; 5]));" },
    },
    BuiltinExample {
        id: "integer-class",
        title: "Preserve a fixed-width integer class",
        program: "ids = uint64([0x0020000000000001u64 0xFFFFFFFFFFFFFFFFu64]);\nT = combinations(ids, [1 2])",
        display_output: Some("T.Var1 is a 4-by-1 uint64 array with exact values"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(T.Var1, \"uint64\"));\nassert(isequal(T.Var1, [ids(1); ids(1); ids(2); ids(2)]));" },
    },
    BuiltinExample {
        id: "empty-input",
        title: "Retain variables for an empty product",
        program: "T = combinations(int16(zeros(0, 1)), [1 2])",
        display_output: Some("T is a 0-by-2 table and T.Var1 remains int16"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(height(T) == 0);\nassert(width(T) == 2);\nassert(isa(T.Var1, \"int16\"));\nassert(isequal(size(T.Var1), [0 1]));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Which input changes fastest?",
        answer: "The rightmost input changes fastest; each input is first read in column-major linear order.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when an input is empty?",
        answer: "The Cartesian product has zero rows. The output table still contains one typed variable for every input.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "nchoosek",
        target: BuiltinDocumentationLinkTarget::Builtin("nchoosek"),
    },
    BuiltinDocumentationLink {
        label: "perms",
        target: BuiltinDocumentationLinkTarget::Builtin("perms"),
    },
    BuiltinDocumentationLink {
        label: "Compatible combinations reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/combinations.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Cartesian-product runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/combinatorics/combinations/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Ordering, containers, integer classes, empty products, limits, and provider policy", location: "crates/runmat-runtime/src/builtins/array/combinatorics/combinations/tests" }],
    notes: &[],
};

pub(super) const COMBINATIONS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("combinations"),
    slug: Some("combinations"),
    summary: "Generate every tuple in the Cartesian product of input arrays.",
    description: "`combinations` constructs a host table whose typed variables enumerate the Cartesian product of the input elements.",
    keywords: &[
        "combinations",
        "cartesian",
        "cartesian product",
        "table",
        "combinatorics",
    ],
    related: &["nchoosek", "perms", "table"],
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
