use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/descriptive/flagged.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("skewness"), slug: Some("skewness"),
    summary: "Compute sample skewness.",
    description: "`skewness(X)` computes the biased third standardized central moment along the first nonsingleton dimension. Flag `0` applies finite-sample bias correction; flag `1` is the default.",
    keywords: &["skewness", "third moment", "statistics", "summary"],
    related: &["kurtosis", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Estimator and missing data", paragraphs: &[
            "The default flag `1` uses the biased moment estimator. Flag `0` applies finite-sample bias correction; slices with fewer than three observations or zero variance return NaN.",
            "NaNs are omitted per slice by default. Use `includenan` to propagate NaN instead. By default, the first nonsingleton dimension is reduced; a scalar dimension, dimension vector, or `\"all\"` can select another reduction.",
            "Real single data yields single output; double data yields double output. Typed integer data and integer controls are separate RunMat-mode extensions. Integer observations enter the double calculation.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Current GPU inputs are gathered through their owning provider for host calculation. GPU `\"all\"` and vector-dimension selectors are RunMat-mode extensions. Direct provider-resident skewness is not yet implemented.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "across-all", title: "Corrected skewness across all elements",
        program: "x = [1 2 3 4]; s = skewness(x, 0, \"all\"); disp(s);",
        display_output: Some("0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(s)); assert(abs(s) < 1e-12);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "What is the default flag?",
        answer: "The default flag is `1`, the biased moment estimator.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "kurtosis", target: BuiltinDocumentationLinkTarget::Builtin("kurtosis") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "MathWorks skewness documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/skewness.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
            label: "Flag, shape, class and missing-value tests",
            location: "crates/runmat-runtime/src/builtins/stats/summary/descriptive/tests/moments.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
