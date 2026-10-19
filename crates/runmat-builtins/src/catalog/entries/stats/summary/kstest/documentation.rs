//! User-facing kstest contract and independently checkable examples.
use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime entry point",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis.rs",
    ),
};
const STATISTICS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "KS statistic and probability calculation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis/kstest_statistics.rs",
    ),
};
const CDF_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "CDF normalization",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis/kstest_cdf.rs",
    ),
};
const ADMISSION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Input and provider admission",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis/kstest_admission.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("kstest"),
    slug: Some("kstest"),
    summary: "Perform a one-sample Kolmogorov-Smirnov test.",
    description: "`kstest(x)` tests whether the numeric vector `x` could come from a standard normal distribution. `[h,p,ksstat,cv] = kstest(...)` returns the logical decision, approximate p-value, KS statistic, and approximate critical value.",
    keywords: &["kstest", "kolmogorov-smirnov", "hypothesis test", "cdf", "statistics"],
    related: &["ttest2", "normcdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Samples and controls", paragraphs: &[
            "`kstest(x)` requires a numeric vector, omits `NaN` observations, and uses the standard normal CDF. At least one non-NaN observation is required.",
            "`Alpha` must be a scalar in (0,1). `Tail` accepts `\"unequal\"`/`\"both\"`, `\"larger\"`/`\"right\"`, and `\"smaller\"`/`\"left\"`.",
            "Native integer samples are a RunMat-only extension, admitted only when every value is exactly representable as binary64. Integer `Alpha` cannot satisfy the required open interval and is rejected.",
        ] },
        BuiltinDocumentationSection { heading: "Reference CDF", paragraphs: &[
            "`CDF` accepts a two-column numeric table of sample points and probabilities in [0,1]. Rows are sorted by sample point; duplicate points collapse to the largest probability. Probabilities must be nondecreasing after normalization, and all sample values must lie within the table's x range. Intermediate values use linear interpolation.",
            "`CDF` also accepts `\"normcdf\"`, `\"normal\"`, `\"norm\"`, `\"gaussian\"`, or an `@normcdf` function handle for the standard normal CDF. Arbitrary function handles and continuous distribution objects are not yet supported.",
            "Native integer CDF tables are a separately gated RunMat extension and reject values that would lose precision during binary64 statistical conversion.",
        ] },
        BuiltinDocumentationSection { heading: "Approximations and residency", paragraphs: &[
            "P-values use large-sample approximations. The approximate critical value is returned for two-sided `Alpha` in [0.01,0.20] or one-sided `Alpha` in [0.005,0.10]; otherwise `cv` is `NaN`.",
            "Automatic internal residency gathers transparently. Explicit `gpuArray` input is a mode-gated RunMat extension and follows the host reference path when enabled. No resident KS kernel or fusion path is claimed.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "normal", title: "Test a centered sample",
            program: "h = kstest([-1 -0.25 0 0.25 1]); disp(h);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(h)); assert(isequal(h,false));" },
        },
        BuiltinExample { id: "one-sided", title: "Request all outputs with a one-sided alternative",
            program: "x = [2 2.5 3 3.5 4]; [h,p,ksstat,cv] = kstest(x, \"Tail\", \"smaller\"); disp(p); disp(ksstat);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(h,true)); assert(p < 0.05); assert(ksstat >= 0 && ksstat <= 1); assert(isscalar(cv));" },
        },
        BuiltinExample { id: "table", title: "Use a tabulated CDF",
            program: "x = [0.2 0.4 0.6 0.8]; cdf = [0 0; 0.5 0.5; 1 1]; [h,p,ksstat,cv] = kstest(x, \"CDF\", cdf); disp(h);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(h,false)); assert(abs(ksstat-0.2)<1e-12); assert(p>=0 && p<=1);" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does kstest accept arbitrary CDF function handles?", answer: "Only `@normcdf` is recognized as a function handle. Use a two-column CDF table for a custom reference distribution." }],
    links: &[
        BuiltinDocumentationLink { label: "ttest2", target: BuiltinDocumentationLinkTarget::Builtin("ttest2") },
        BuiltinDocumentationLink { label: "normcdf", target: BuiltinDocumentationLinkTarget::Builtin("normcdf") },
        SOURCE,
        STATISTICS_SOURCE,
        CDF_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, STATISTICS_SOURCE, CDF_SOURCE, ADMISSION_SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CDF, tail, integer, and admission tests", location: "crates/runmat-runtime/src/builtins/stats/summary/hypothesis/tests/kstest.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Registered-provider admission, causes, and borrowed lifetime tests", location: "crates/runmat-runtime/src/builtins/stats/summary/hypothesis/kstest_provider_tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "One-sample KS execution in MATLAB-syntax code", location: "crates/runmat-vm/tests/stats_ml.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Integer KS observations and exact conversion boundary", location: "crates/runmat-vm/tests/integer_json_join_and_statistical_semantics.rs" },
        ],
        notes: &["Resident observations and CDF tables are materialized through their owning provider for the host calculation."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
