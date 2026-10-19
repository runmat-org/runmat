use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("geomean"), slug: Some("geomean"),
    summary: "Compute the geometric mean of nonnegative observations.",
    description: "`geomean(X)` reduces the first non-singleton dimension. Use a dimension, a dimension vector, or `\"all\"` to select other axes. `\"includenan\"` is the default; `\"omitnan\"` excludes NaNs from each slice.",
    keywords: &["geomean", "geometric mean", "statistics", "summary"],
    related: &["harmmean", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input and result class", paragraphs: &[
            "The documented data domain is real, nonnegative single or double. Typed integer data is a RunMat-mode extension; the result is double for integer data and retains single or double for floating data.",
            "The selected dimensions become singleton dimensions. An empty reduction slice produces NaN; `\"omitnan\"` produces NaN when no observations remain.",
            "A zero produces a zero result for its slice unless infinity is also present, in which case the result is NaN. A negative observation produces NaN only in its affected output slice.",
            "Selecting only singleton or trailing dimensions leaves nonnegative input values unchanged; negative observations still produce NaN.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The host implementation reduces each selected slice. Eligible floating GPU arrays enter an owner-checked resident provider route; unsupported provider forms preserve the documented host-materialization behavior for GPU inputs. Resident execution and performance qualification remain in progress.",
        ]},
    ],
    examples: &[BuiltinExample {
        id: "column-geometric-means", title: "Column geometric means",
        program: "m = geomean([1 9; 4 16])",
        display_output: Some("m = [2 12]"), compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(m-[2 12])) < 1e-12);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "How do I ignore missing values?",
        answer: "Pass `\"omitnan\"`, for example `geomean(X,1,\"omitnan\")`.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "harmmean", target: BuiltinDocumentationLinkTarget::Builtin("harmmean") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/geomean.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/geomean.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistical reduction classes, values and residency", location: "crates/runmat-runtime/src/builtins/stats/summary/statistical_means/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
