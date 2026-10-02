//! Public ECDF behavior, examples, and implementation evidence.
use crate::*;

const OWNER: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Empirical distribution runtime",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/ecdf.rs"),
};
const OPTIONS: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Typed ECDF control admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/ecdf/options.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ecdf"),
    slug: Some("ecdf"),
    summary: "Compute or plot empirical distribution functions.",
    description: "`ecdf(y)` computes the empirical cumulative distribution function for numeric sample data. With no requested outputs it plots the empirical function; with outputs it returns column vectors for function values, evaluation points, and confidence bounds.",
    keywords: &["ecdf", "empirical cdf", "survivor", "cumulative hazard", "statistics"],
    related: &["cdfplot", "stairs"],
    sections: &[
        BuiltinDocumentationSection { heading: "Observations and controls", paragraphs: &[
            "NaN sample, censoring, and frequency entries are omitted. `Frequency` supplies nonnegative integer observation counts.",
            "`Function` accepts `\"cdf\"`, `\"survivor\"`, and `\"cumulative hazard\"`; `\"cumhazard\"` and `\"hazard\"` are accepted aliases. Fully observed and right-censored cumulative hazard uses Nelson–Aalen steps.",
            "`Censoring` supports fully observed (`0`), left-censored (`-1`), and right-censored (`1`) observations. Interval and double-censored Turnbull estimation remain unsupported rather than approximated.",
            "`Alpha` sets pointwise confidence bounds. `\"Bounds\",\"on\"` includes confidence-bound lines when plotting. Interval-censored iteration controls return a stable unsupported-form error.",
        ] },
        BuiltinDocumentationSection { heading: "Classes, compatibility, and placement", paragraphs: &[
            "Typed-integer or logical sample data, typed-integer or logical `Frequency` data, and typed-integer `Censoring` data are separately named RunMat-only extensions. MATLAB compatibility mode rejects each role before provider access.",
            "Admitted integer values are checked for exact binary64 conversion at the statistics boundary. Numeric results are double; explicit conversion is required for an integer value that cannot cross that boundary exactly.",
            "Floating resident samples gather through their owning provider for host statistics. Numeric outputs are uploaded back to that provider; plotting remains a host graphics sink. This path does not execute a resident ECDF kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "empirical-cdf", title: "Compute an empirical CDF",
            program: "[f,x] = ecdf([3;1;2;2]); disp(f); disp(x);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(f,[0;0.25;0.75;1])); assert(isequal(x,[1;1;2;3]));" },
        },
        BuiltinExample {
            id: "weighted-survivor", title: "Compute a survivor function with frequencies",
            program: "[s,x] = ecdf([1;2;3], \"Frequency\", [2;1;1], \"Function\", \"survivor\"); disp(s);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(s,[1;0.5;0.25;0])); assert(isequal(x,[1;1;2;3]));" },
        },
        BuiltinExample {
            id: "plot-empirical-cdf", title: "Plot an empirical CDF",
            program: "ecdf([1;2;2;3]);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(get(gca,'Children'))>=1);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "cdfplot", target: BuiltinDocumentationLinkTarget::Builtin("cdfplot") },
        BuiltinDocumentationLink { label: "stairs", target: BuiltinDocumentationLinkTarget::Builtin("stairs") },
        OWNER,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[OWNER, OPTIONS],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Empirical CDF, censoring, integer, plotting, and owner-boundary tests", location: "crates/runmat-runtime/src/builtins/stats/summary/ecdf/tests.rs" }],
        notes: &["Host evaluation follows owner-aware gather; plot state is created only for no-output forms."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
