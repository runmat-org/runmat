use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/descriptive/flagged.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("kurtosis"), slug: Some("kurtosis"),
    summary: "Compute sample kurtosis.",
    description: "`kurtosis(X)` computes biased sample kurtosis, not excess kurtosis. Flag `0` applies finite-sample bias correction; flag `1` is the default.",
    keywords: &["kurtosis", "fourth moment", "statistics", "summary"],
    related: &["skewness", "std"],
    sections: &[
        BuiltinDocumentationSection { heading: "Estimator and missing data", paragraphs: &[
            "The default flag `1` uses the biased moment estimator. Flag `0` applies finite-sample bias correction; slices with fewer than four observations or zero variance return NaN.",
            "NaNs are omitted per slice by default. Use `includenan` to propagate NaN instead. By default, the first nonsingleton dimension is reduced; a scalar dimension, dimension vector, or `\"all\"` can select another reduction.",
            "Real single data yields single output; double data yields double output. Typed integer data and integer controls are separate RunMat-mode extensions. Integer observations enter the double calculation.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Current GPU inputs are gathered through their owning provider for host calculation. GPU `\"all\"` and vector-dimension selectors are RunMat-mode extensions. Direct provider-resident kurtosis is not yet implemented.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "across-all", title: "Kurtosis across all elements",
        program: "x = [1 2 3 4]; k = kurtosis(x, 1, \"all\"); disp(k);",
        display_output: Some("1.6400"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(k)); assert(abs(k-1.64) < 1e-12);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does kurtosis return excess kurtosis?",
        answer: "No. It returns ordinary kurtosis; the normal-distribution reference value is 3, not 0.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "skewness", target: BuiltinDocumentationLinkTarget::Builtin("skewness") },
        BuiltinDocumentationLink { label: "std", target: BuiltinDocumentationLinkTarget::Builtin("std") },
        BuiltinDocumentationLink { label: "MathWorks kurtosis documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/kurtosis.html") },
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
