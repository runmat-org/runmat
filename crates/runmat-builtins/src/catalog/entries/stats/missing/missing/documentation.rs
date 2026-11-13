use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("missing"),
    slug: Some("missing"),
    summary: "Create missing string scalars or arrays.",
    description: "`missing` creates a 1-by-1 string array containing a typed missing element, displayed as `<missing>`. The documented MATLAB function takes no input arguments.",
    keywords: &["missing", "string", "missing values"],
    related: &["ismissing", "strings", "fillmissing", "rmmissing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Size forms and compatibility",
            paragraphs: &[
                "`missing(n)`, `missing(m,n,...)`, and `missing(sz)` are gated RunMat conveniences. A single scalar size constructs an n-by-n array; a size vector supplies its element values as dimensions. An empty size vector creates a 0-by-0 array.",
                "All eight fixed-width integer size classes are decoded exactly and checked against nonnegative host allocation limits. MATLAB-compatible mode rejects every argument before gathering a resident size control or consulting its provider.",
                "The result uses host StringArray storage and displays each missing element as `<missing>`. A supported `like` prototype in the RunMat-only sized form does not change the output class or device.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "scalar",
            title: "Create a missing scalar",
            program: "S = missing",
            display_output: Some("S = <missing>"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(S), [1 1])); assert(ismissing(S));",
            },
        },
        BuiltinExample {
            id: "shaped",
            title: "Create a RunMat missing string array",
            program: "S = missing(uint16(2), uint8(3))",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(S), [2 3])); assert(all(ismissing(S), 'all'));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ismissing", target: BuiltinDocumentationLinkTarget::Builtin("ismissing") },
        BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
        BuiltinDocumentationLink { label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing") },
        BuiltinDocumentationLink { label: "rmmissing", target: BuiltinDocumentationLinkTarget::Builtin("rmmissing") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/detection.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "String-array construction and exact sizes", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/detection.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, shaped, exact integer and compatibility admission", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/core.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing construction", location: "crates/runmat-vm/tests/missing_values.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Exact integer size controls and MATLAB-mode rejection", location: "crates/runmat-vm/tests/integer_datetime_metadata_and_nonzero_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
