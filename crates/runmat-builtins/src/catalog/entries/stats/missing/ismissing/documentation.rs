use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "typed-string-presence",
        title: "Distinguish missing strings from literal spellings",
        program: "values = [string(missing), \"<missing>\", \"<MISSING>\", \"\"];\nmask = ismissing(values)",
        display_output: Some("mask = [true false false false]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(mask, logical([1 0 0 0]))); assert(isequal(values(2), \"<missing>\")); assert(isequal(values(3), \"<MISSING>\"));",
        },
    },
    BuiltinExample {
        id: "nan-values",
        title: "Detect NaN values",
        program: "tf = ismissing([1 NaN 3])",
        display_output: Some("tf = [false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 0])));",
        },
    },
    BuiltinExample {
        id: "table-values",
        title: "Detect missing table entries",
        program: "T = table([1; NaN], [2; 3], 'VariableNames', {'A','B'});\nM = ismissing(T)",
        display_output: Some("M = [false false; true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(M, logical([0 0; 1 0])));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ismissing"),
    slug: Some("ismissing"),
    summary: "Return a logical mask identifying missing entries.",
    description: "`ismissing(A)` identifies missing entries in supported values. Numeric and complex entries are missing when a stored component is NaN; strings are missing only when their elements carry typed absence, not when their text spells `<missing>`.",
    keywords: &["ismissing", "missing", "NaN", "NaT", "string", "table"],
    related: &["missing", "anymissing", "fillmissing", "rmmissing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Containers and calendar values",
            paragraphs: &[
                "Character-array rows are missing when blank; literal `<missing>` characters are ordinary text. Cells are tested element by element. Structures retain their fields with recursively classified values. Tables return one logical column per table variable, ordered as the table variables are stored.",
                "Datetime and duration objects are missing where their stored numeric calendar or duration representation is NaN. Fixed-width integer and logical arrays have no standard missing element; they produce false masks without floating conversion.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Resident input",
            paragraphs: &[
                "An explicitly GPU-resident input is a RunMat-only form and is rejected by MATLAB compatibility mode before provider access. Admitted resident integers use validated owner and class metadata to build an all-false host mask without a payload download. Other resident values gather recursively through their owners before inspection.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "missing", target: BuiltinDocumentationLinkTarget::Builtin("missing") },
        BuiltinDocumentationLink { label: "anymissing", target: BuiltinDocumentationLinkTarget::Builtin("anymissing") },
        BuiltinDocumentationLink { label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing") },
        BuiltinDocumentationLink { label: "rmmissing", target: BuiltinDocumentationLinkTarget::Builtin("rmmissing") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/detection.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Missing-value detection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/detection.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric, integer, structure, calendar and table forms", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/core.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value behavior", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
