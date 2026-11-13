//! Public CDF plot forms and scene/statistics examples.
use crate::*;

const OWNER: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "CDF plot runtime",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/ecdf.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cdfplot"), slug: Some("cdfplot"),
    summary: "Plot an empirical cumulative distribution function.",
    description: "`cdfplot(x)` plots the empirical cumulative distribution function for a single- or double-precision numeric vector. At a value `t`, the empirical CDF is the fraction of non-NaN observations less than or equal to `t`.",
    keywords: &["cdfplot", "empirical cdf", "plot", "statistics"],
    related: &["ecdf", "plot"],
    sections: &[
        BuiltinDocumentationSection { heading: "Outputs and sample classes", paragraphs: &[
            "`h` is the line handle. `stats` is a host structure with `min`, `max`, `mean`, `median`, and sample `std` fields computed after omitting NaN observations.",
            "Dense typed-integer and dense logical samples are RunMat extensions. Integer values must lie within the contiguous exact binary64 integer range at the deliberate statistics and plotting boundary.",
            "Use `ecdf` for numeric ECDF outputs, confidence bounds, frequency weights, survivor functions, and right-censored data.",
        ] },
        BuiltinDocumentationSection { heading: "Graphics and resident inputs", paragraphs: &[
            "Resident sample data is a RunMat extension. The owning provider supplies host data for the plot; the line handle and statistics remain host outputs. No resident CDF plot kernel is claimed.",
            "The call creates or updates a host figure and may notify graphics observers. It requires a desktop or browser graphics host.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "cdfplot-stats", title: "Plot an empirical CDF and read summary statistics",
        program: "[h,stats] = cdfplot([1;2;2;3]); disp(stats.mean);",
        display_output: Some("2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(h>0); assert(stats.min==1); assert(stats.max==3); assert(stats.mean==2); assert(stats.median==2); assert(abs(stats.std-sqrt(2/3))<1e-12); assert(numel(get(h,'XData'))>=4); y = get(h,'YData'); assert(y(end)==1);" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ecdf", target: BuiltinDocumentationLinkTarget::Builtin("ecdf") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        OWNER,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[OWNER],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CDF line, summary fields, input classes and host plotting", location: "crates/runmat-runtime/src/builtins/stats/summary/ecdf/tests.rs" }],
        notes: &["Host-side plot and summary publication; resident input gathers through the owner."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
