use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Frequency-table implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/descriptive/tabulation.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tabulate"), slug: Some("tabulate"),
    summary: "Create a frequency table.",
    description: "`tabulate(X)` reports observed values, counts, and percentages in three columns. Real floating-point observations return a numeric matrix. Typed integers, logical arrays, and text labels return a cell table so their labels retain their class or exact value.",
    keywords: &["tabulate", "frequency", "count", "percent", "statistics"],
    related: &["histcounts", "mode"],
    sections: &[
        BuiltinDocumentationSection { heading: "Grouping and percentages", paragraphs: &[
            "Positive whole-number floating inputs include zero-count levels from 1 through the greatest level, up to 1,000,000 levels. A larger floating level is rejected before constructing the table. Other real values are sorted and grouped by exact value. NaNs are excluded from both the table and percentage denominator.",
            "Integer observations of all eight classes group without conversion through double. Positive-level domains up to 1,000,000 levels include missing levels; larger domains report observed integer levels. Counts and percentages are double, while integer labels remain exact in the first cell column, including `uint64` values above `flintmax`.",
            "Logical arrays, string arrays, and cell arrays of scalar labels return a cell table. Character rows are treated as a sequence of character labels. Scalar logical values follow the scalar numeric path.",
        ] },
        BuiltinDocumentationSection { heading: "Residency and compatibility", paragraphs: &[
            "Resident inputs are gathered through their owning provider and tabulated on the host. This is an intentional host-materialization boundary, not a resident frequency kernel. GPU input and typed-integer input are separate RunMat-only extensions; MATLAB compatibility mode rejects those forms before materialization.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "levels", title: "Include an unobserved positive level",
            program: "tbl = tabulate([1 3 3]); disp(tbl);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tbl(:,1), [1;2;3])); assert(isequal(tbl(:,2), [1;0;2])); assert(abs(sum(tbl(:,3))-100)<1e-10);" },
        },
        BuiltinExample { id: "missing", title: "Omit NaN observations",
            program: "tbl = tabulate([1 NaN 1 2]); disp(tbl);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tbl(:,1:2),[1 2;2 1])); assert(abs(tbl(1,3)-200/3)<1e-10);" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Why does tabulate([1 3 3]) include 2?",
        answer: "A small positive whole-number domain includes the missing level with frequency zero.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "histcounts", target: BuiltinDocumentationLinkTarget::Builtin("histcounts") },
        BuiltinDocumentationLink { label: "mode", target: BuiltinDocumentationLinkTarget::Builtin("mode") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, BuiltinDocumentationLink { label: "Descriptive statistics and provider materialization", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/descriptive.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Numeric, integer, logical, text, missing, and provider-boundary tests",
            location: "crates/runmat-runtime/src/builtins/stats/summary/descriptive/tests/tabulation.rs",
        }],
        notes: &["Tabulation runs on the host after resident input materialization; borrowed provider allocations remain owned by their caller."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
