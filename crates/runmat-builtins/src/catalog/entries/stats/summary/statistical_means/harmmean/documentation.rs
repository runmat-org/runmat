use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("harmmean"), slug: Some("harmmean"),
    summary: "Compute the harmonic mean of real observations.",
    description: "`harmmean(X)` reduces the first non-singleton dimension. A dimension, dimension vector, or `\"all\"` selects other axes. `\"includenan\"` is the default; `\"omitnan\"` ignores NaNs in each slice.",
    keywords: &["harmmean", "harmonic mean", "statistics", "summary"],
    related: &["geomean", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior and classes", paragraphs: &[
            "The harmonic mean is the count of observations divided by the sum of their reciprocals. A zero observation produces zero; an empty or all-omitted slice produces NaN.",
            "Selecting only singleton or trailing dimensions leaves the input values unchanged.",
            "Real single and double are documented data classes. Typed integer observations are a RunMat-mode extension and return double; floating inputs retain their class.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Eligible floating GPU arrays enter an owner-checked resident provider route using scaled reciprocals. Unsupported provider forms preserve the documented host-materialization behavior for GPU inputs. Resident performance and edge-case qualification remain in progress.",
        ]},
    ],
    examples: &[BuiltinExample {
        id: "all-elements", title: "Harmonic mean across all values",
        program: "m = harmmean([1 2 4], \"all\")",
        display_output: Some("m = 12/7"), compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(abs(m-12/7) < 1e-12);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Can I reduce over several dimensions?",
        answer: "Yes. Pass a dimension vector such as `harmmean(X,[1 3])`.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "geomean", target: BuiltinDocumentationLinkTarget::Builtin("geomean") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/harmmean.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/harmmean.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistical reduction classes, values and residency", location: "crates/runmat-runtime/src/builtins/stats/summary/statistical_means/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
