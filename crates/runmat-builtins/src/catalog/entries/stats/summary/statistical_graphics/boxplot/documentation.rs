//! Boxplot documentation preserves the legacy public field intent and examples.
use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot.rs"),
};
const ADMISSION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot class and shape admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/admission.rs"),
};
const OPTIONS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot ordered option plan",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/plan.rs"),
};
const OPTION_DECODING_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot option value decoding",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/plan/decoding.rs"),
};
const GROUPS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot grouping and statistics",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/groups.rs"),
};
const LABELS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot exact group identity",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/groups/labels.rs"),
};
const PLOT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot graphics rendering",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/plot.rs"),
};
const APPEARANCE_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boxplot typed appearance and notch geometry",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/boxplot/appearance.rs"),
};
const ECDF: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "ecdf",
    target: BuiltinDocumentationLinkTarget::Builtin("ecdf"),
};
const HISTOGRAM: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "histogram",
    target: BuiltinDocumentationLinkTarget::Builtin("histogram"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("boxplot"),
    slug: Some("boxplot"),
    summary: "Visualize sample quartiles, whiskers, medians, and outliers.",
    description: "`boxplot(x)` plots one box for a vector and one box per matrix column. `boxplot(x,g)` groups vector data by numeric, text, cell, or categorical grouping variables. With an output, RunMat returns a row vector of graphics handles for the line, scatter, and filled-patch primitives used to draw the boxes. Public numeric data, groups, and numeric option values are single or double; integer and logical forms are independently gated RunMat extensions.",
    keywords: &["boxplot", "box plot", "statistics", "quartile", "outlier", "visualization"],
    related: &["ecdf", "histogram"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Samples, groups, and quartiles",
            paragraphs: &[
                "NaN data values are omitted from quartile, median, whisker, and outlier calculations. Quartiles use linear interpolation over sorted finite observations; whiskers extend to the most extreme observations within `Whisker` times the interquartile range.",
                "Grouping variables can be numeric vectors, string arrays, character matrices, cell arrays of scalar group labels, categorical arrays, or cell arrays of grouping vectors. Missing group labels are omitted.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Options and rendering",
            paragraphs: &[
                "`DataLim`, `ExtremeMode`, and `Jitter` control outlier display. `ExtremeMode` clips or compresses outlier markers outside the data limits, and traditional plots default to no jitter.",
                "Recognized options include `Labels`, `GroupOrder`, `Orientation`, `Positions`, `Widths`, `Whisker`, `Colors`, `ColorGroup`, `Symbol`, `OutlierSize`, `PlotStyle`, `BoxStyle`, `MedianStyle`, `Notch`, `DataLim`, `ExtremeMode`, `Jitter`, `LabelOrientation`, `LabelVerbosity`, and factor display validation options.",
                "`BoxStyle` selects an outline or filled box; `MedianStyle` selects a line or target marker. `Notch` displays the median comparison interval as notches in traditional plots or triangle markers in compact plots. The interval endpoints are the median plus or minus 1.57 times the interquartile range divided by the square root of the non-NaN observation count. `ColorGroup` and factor display layout options are admitted but do not yet change rendered colors or layout.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Numerical classes and residency",
            paragraphs: &[
                "In RunMat mode, all eight integer classes and logical arrays are accepted independently for sample data, grouping data, and numeric option values. Integer sample observations and visualization-valued numeric options cross an explicit binary64 boundary; integer grouping identity, numeric order, and numeric labels remain exact, including `uint64` values above `flintmax`.",
                "Interactive `gpuArray` input is an independently gated RunMat extension. After compatibility admission, RunMat gathers to the host before statistics and rendering; graphics handles remain host-resident, without claiming a GPU reduction kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "boxplot-columns", title: "Plot one box for each matrix column",
            program: "h = boxplot([1 10; 2 20; 3 30; 4 40])",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(h)>0);" },
        },
        BuiltinExample {
            id: "boxplot-grouped", title: "Plot grouped data with labels",
            program: "h = boxplot([1;2;10;20], {'A';'A';'B';'B'}, 'Labels', {'First','Second'})",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(h)>0);" },
        },
        BuiltinExample {
            id: "boxplot-horizontal", title: "Use horizontal orientation and a custom whisker length",
            program: "h = boxplot([1;2;3;100], 'Orientation', 'horizontal', 'Whisker', 1)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(h)>0);" },
        },
        BuiltinExample {
            id: "boxplot-filled-comparison-interval", title: "Draw a filled box and median comparison markers",
            program: "h = boxplot([1;2;3;4;5], 'BoxStyle', 'filled', 'MedianStyle', 'target', 'Notch', 'marker')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(h)>=8);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[SOURCE, ECDF, HISTOGRAM],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ADMISSION_SOURCE, OPTIONS_SOURCE, OPTION_DECODING_SOURCE, GROUPS_SOURCE, LABELS_SOURCE, PLOT_SOURCE, APPEARANCE_SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Box grouping, quartiles, options, compatibility and plotted handles", location: "crates/runmat-runtime/src/builtins/stats/summary/boxplot/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident admission and borrowed input ownership", location: "crates/runmat-runtime/src/builtins/stats/summary/boxplot/provider_tests.rs" },
        ],
        notes: &["Plot geometry is computed on the host and published through the graphics system."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
