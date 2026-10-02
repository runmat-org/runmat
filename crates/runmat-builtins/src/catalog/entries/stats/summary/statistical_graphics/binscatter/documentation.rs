//! Public binned-scatter behavior and executable chart examples.
use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Binned scatter implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/binscatter.rs",
    ),
};
const OPTIONS: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Binned-scatter options", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/binscatter/options.rs") };
const INPUT: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Binned-scatter input admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/binscatter/input.rs") };
const BINNING: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Binned-scatter numeric binning", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/binscatter/numeric.rs") };
const PLOT: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Binned-scatter chart rendering", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/binscatter/plot.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("binscatter"),
    slug: Some("binscatter"),
    summary: "Visualize point density with a two-dimensional binned scatter plot.",
    description: "`binscatter(x,y)` bins paired observations and displays counts as colored tiles. A scalar or two-element `nbins` controls the number of bins. Table/timetable forms select numeric variables. Name-value controls include `NumBins`, `XLimits`, `YLimits`, `ShowEmptyBins`, `FaceAlpha`, `DisplayName`, and `DisplayStyle` set to `\"tile\"`.",
    keywords: &["binscatter", "binned scatter", "2d histogram", "statistics", "plotting", "density"],
    related: &["histcounts2", "scatter", "histogram2"],
    sections: &[
        BuiltinDocumentationSection { heading: "Binning and chart properties", paragraphs: &[
            "Default bin counts use Scott-rule estimates capped at 100 bins per dimension. Explicit `NumBins` values are positive integers no greater than 250. Empty bins are hidden from the rendered tile surface by default but remain in the `Values` matrix; `ShowEmptyBins` reveals them.",
            "The returned handle has class `binscatter`. `get` exposes `Values`, `XData`, `YData`, `NumBins`, `XBinEdges`, `YBinEdges`, `XLimits`, `YLimits`, `ShowEmptyBins`, `FaceAlpha`, and `DisplayName`. `set` recomputes the chart after `NumBins`, `ShowEmptyBins`, `XLimits`, `YLimits`, `XData`, `YData`, `FaceAlpha`, or `DisplayName` changes.",
            "RunMat renders an indexed image surface with a colorbar. MATLAB's interactive chart UI affordances are not modeled.",
        ] },
        BuiltinDocumentationSection { heading: "Numerical classes and residency", paragraphs: &[
            "Real X/Y vectors, bin counts and explicit axis limits accept all eight built-in integer classes. Integer data and explicit limits retain their native class and exact chart-state values. Bin membership remains exact across the uint64 range when an axis has integer data and either integer explicit limits or no explicit limits; floating limits use a binary64 projection. Bin edges are displayed as floating-point values.",
            "Resident GPU-array input is a RunMat-only gather extension. MATLAB-compatible mode rejects resident input before provider access; the returned chart handle and rendering state are host-resident, not a claim of GPU binning.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "binscatter-counts", title: "Count points in four tiles",
            program: "h = binscatter([0; 0.2; 0.8; 1], [0; 0.1; 0.9; 1], [2 2]); values = get(h,'Values'); get(h,'Values')",
            display_output: Some("ans =\n     2     0\n     0     2"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(values,[2 0;0 2])); assert(isequal(get(h,'NumBins'),[2 2]));" },
        },
        BuiltinExample {
            id: "binscatter-limits", title: "Set limits and reveal empty tiles",
            program: "x = [0; 0.2; 0.8; 1]; y = [-1; -0.8; 0.8; 1]; h = binscatter(x,y,'NumBins',[20 10],'XLimits',[0 1],'YLimits',[-1 1],'ShowEmptyBins','on'); get(h,'NumBins')",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(get(h,'NumBins'),[20 10])); assert(isequal(get(h,'XLimits'),[0 1])); assert(isequal(get(h,'YLimits'),[-1 1])); assert(get(h,'ShowEmptyBins')); counts = get(h,'Values'); assert(isequal(size(counts),[20 10])); assert(sum(counts(:)) == 4);" },
        },
        BuiltinExample {
            id: "binscatter-table", title: "Select table variables",
            program: "x = [0; 0.2; 0.8; 1]; y = [0; 0.1; 0.9; 1]; T = table(x,y); h = binscatter(T,'x','y',25); get(h,'NumBins')",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(get(h,'NumBins'),[25 25])); counts = get(h,'Values'); assert(isequal(size(counts),[25 25])); assert(sum(counts(:)) == 4);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does hiding empty tiles change Values?", answer: "No. `ShowEmptyBins` changes visibility, while `Values` retains the numeric count for every bin." },
    ],
    links: &[SOURCE, OPTIONS, INPUT, BINNING, PLOT],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, OPTIONS, INPUT, BINNING, PLOT],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact integer bins, chart state, get/set and graphics handles", location: "crates/runmat-runtime/src/builtins/stats/summary/binscatter/tests.rs" }],
        notes: &["Indexed image surface and colorbar render through the host graphics system."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
