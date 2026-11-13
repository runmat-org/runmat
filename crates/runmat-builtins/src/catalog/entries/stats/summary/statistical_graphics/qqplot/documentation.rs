//! Public Q-Q plot behavior, limits and executable graphics examples.
use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Q-Q plot implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/qqplot.rs",
    ),
};
const ADMISSION: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Q-Q input admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/qqplot/admission.rs") };
const SERIES: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Q-Q quantile series", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/qqplot/series.rs") };
const GEOMETRY: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Q-Q reference geometry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/qqplot/geometry.rs") };
const PLOT: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Q-Q rendering", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/qqplot/plot.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("qqplot"),
    slug: Some("qqplot"),
    summary: "Create a quantile-quantile plot against a normal distribution or another sample.",
    description: "`qqplot(x)` plots sorted sample quantiles against standard-normal theoretical quantiles. `qqplot(x,y)` compares the quantiles of two samples. `qqplot(x,[],pvec)` and `qqplot(x,y,pvec)` evaluate quantiles at explicit percentage values in [0,100]. Matrix inputs create one series per column, omitting NaN and Inf observations.",
    keywords: &["qqplot", "quantile", "normal", "probability", "statistics", "plotting"],
    related: &["cdfplot", "ecdf"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Samples and returned handles",
            paragraphs: &[
                "For each plotted series, the returned handle vector contains the quantile-point line, a solid quartile line, and a dashed fitted reference line. A one-sample plot returns a 3-by-1 vector of line handles. The one-sample reference passes through the first and third quartiles of the theoretical normal and observed sample quantiles.",
                "RunMat supports numeric vectors and matrices. Two sample matrices are paired by column; one vector can be compared with each column of a matrix. Distribution-object inputs and table-variable routing are outside this supported surface.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and residency",
            paragraphs: &[
                "The compatibility target documents single and double samples and percentages. RunMat mode also accepts the eight typed-integer classes after exact binary64 representation checks, and logical numeric inputs as a separately gated extension. Strict compatibility mode rejects these extensions.",
                "Automatically resident inputs gather at the host quantile-and-graphics boundary. Explicit `gpuArray` inputs require a separate RunMat extension gate. This function does not promise GPU-resident quantile calculation or GPU-resident output handles.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "qqplot-normal",
            title: "Compare a sample with a normal distribution",
            program: "h = qqplot([1; 2; 4; 8])",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(numel(h)==3); assert(isequal(get(h(1),'YData'),[1 2 4 8]));",
            },
        },
        BuiltinExample {
            id: "qqplot-percentages",
            title: "Compare explicit sample quantiles (YData is [5 10 15])",
            program: "h = qqplot([0; 10; 20], [], [25 50 75]); y = get(h(1),'YData')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(y,[5 10 15]));",
            },
        },
        BuiltinExample {
            id: "qqplot-two-samples",
            title: "Compare median quantiles 3 and 4",
            program: "h = qqplot([1; 3; 5], [2; 4; 6], 50)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(get(h(1),'XData'),3)); assert(isequal(get(h(1),'YData'),4));",
            },
        },
        BuiltinExample {
            id: "qqplot-invalid-percentage",
            title: "Reject percentages outside the closed interval",
            program: "qqplot([0; 10; 20], [], 101)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:qqplot:InvalidArgument",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does the reference line show?",
            answer: "The line relates the first and third quartiles of theoretical and observed quantiles; it is not a second sample series.",
        },
    ],
    links: &[SOURCE, ADMISSION, SERIES, GEOMETRY, PLOT],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ADMISSION, SERIES, GEOMETRY, PLOT],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Normal/two-sample quantile geometry, integer extension gates and graphics handles",
            location: "crates/runmat-runtime/src/builtins/stats/summary/qqplot/tests.rs",
        }],
        notes: &["Host graphics result; resident inputs use the checked host quantile boundary."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
