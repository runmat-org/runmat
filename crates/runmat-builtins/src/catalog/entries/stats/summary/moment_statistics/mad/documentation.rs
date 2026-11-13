use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/descriptive/flagged.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mad"), slug: Some("mad"),
    summary: "Compute mean or median absolute deviation.",
    description: "`mad(X)` computes mean absolute deviation from the mean. Pass flag `1` to compute median absolute deviation from the median.",
    keywords: &["mad", "mean absolute deviation", "median absolute deviation", "statistics"],
    related: &["mean", "median"],
    sections: &[
        BuiltinDocumentationSection { heading: "Deviation and dimensions", paragraphs: &[
            "Flag `0` (the default) calculates `mean(abs(X-mean(X)))`; flag `1` calculates `median(abs(X-median(X)))`.",
            "By default, `mad` reduces the first nonsingleton dimension. A scalar dimension, dimension vector, or `\"all\"` selects another reduction. Each empty slice returns NaN; NaNs are omitted from each slice by default. `includenan` preserves their contribution as NaN.",
            "Real single inputs produce single outputs; double inputs produce double outputs. Real typed-integer data and typed-integer controls are separately gated RunMat extensions. Integer data enters a double calculation and returns double.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The current deviation calculation runs on the host and gathers resident input through its owning provider. It is a fusion boundary and does not mutate its input. Direct provider-resident reduction is not yet implemented.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "mean-default", title: "Mean absolute deviation by default",
            program: "d = mad([1 2 10], 0, \"all\"); fprintf('d = %.4f\\n', d);",
            display_output: Some("d = 3.7778"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(d)); assert(abs(d-34/9)<1e-10);" },
        },
        BuiltinExample { id: "median", title: "Median absolute deviation",
            program: "d = mad([1 2 10], 1, \"all\"); fprintf('d = %.4f\\n', d);",
            display_output: Some("d = 1.0000"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(d)); assert(d==1);" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "What is the default deviation center?",
        answer: "`mad(X)` uses the arithmetic mean. Pass flag `1` to use the median as the center.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "median", target: BuiltinDocumentationLinkTarget::Builtin("median") },
        BuiltinDocumentationLink { label: "MathWorks mad documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/mad.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
            label: "Mean/median flags, class, dimension, integer boundary and missing-value tests",
            location: "crates/runmat-runtime/src/builtins/stats/summary/descriptive/tests/moments.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
