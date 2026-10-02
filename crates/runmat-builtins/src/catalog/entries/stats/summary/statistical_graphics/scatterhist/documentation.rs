//! Scatterhist documentation retains the legacy forms and visible examples.
use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Scatterhist entrypoint", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist.rs") };
const ADMISSION: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Input and provider admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist/admission.rs") };
const CONTROLS: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Numeric and layout controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist/controls.rs") };
const STYLES: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Group styles", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist/styles.rs") };
const LABELS: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Group labels", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist/labels.rs") };
const NUMERIC: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Histogram and density computation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist/numeric.rs") };
const RENDER: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Chart rendering", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/scatterhist/render.rs") };
const MATLAB: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "MATLAB scatterhist public reference",
    target: BuiltinDocumentationLinkTarget::External(
        "https://www.mathworks.com/help/stats/scatterhist.html",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("scatterhist"),
    slug: Some("scatterhist"),
    summary: "Create a scatter plot with marginal histograms.",
    description: "`scatterhist(x,y)` creates a scatter plot with marginal distributions for vector inputs. `scatterhist(x,y,Name,Value)` accepts `NBins`, `Group`, `PlotGroup`, `Kernel`, `Style`, `Bandwidth`, `Legend`, `Location`, `Direction`, `Color`, `Marker`, `MarkerSize`, `LineStyle`, `LineWidth`, and `Parent`. An output is a 3-by-1 vector of scatter, X-marginal, and Y-marginal axes handles.",
    keywords: &["scatterhist", "scatter", "histogram", "marginal", "statistics", "plotting"],
    related: &["scatter", "histogram"],
    sections: &[
        BuiltinDocumentationSection { heading: "Bins and grouped distributions", paragraphs: &[
            "RunMat composes a 2-by-2 subplot grid; MATLAB's exact linked-axes positioning and interactive UI are not modeled. Default bin counts use Scott's rule capped at 100 per marginal. Explicit `NBins` entries are integers from 2 through 250.",
            "`Group` accepts numeric, logical, character-matrix, string, cell, or categorical labels. A group creates its own scatter and marginal series. A grouped chart defaults to a legend and stair outlines; `PlotGroup` set to `off` pools marginal observations. `Style` selects `bar` or `stairs`.",
            "`Kernel` accepts `off`, `on`, or `overlay`; the latter draws density lines over histogram counts. `Bandwidth` supplies positive X/Y bandwidths. The documented 2-by-K matrix puts X bandwidths in its first row and Y bandwidths in its second; RunMat also retains its earlier K-by-2 per-group layout when K differs from two. A 2-by-2 matrix follows the documented row-axis layout.",
        ] },
        BuiltinDocumentationSection { heading: "Layout, styling and numerical classes", paragraphs: &[
            "`Location` defaults to `southwest`; `Direction` defaults to `in` and reverses the marginal count or density direction when set to `out`. `Color`, `Marker`, `MarkerSize`, `LineStyle`, `LineWidth`, and `Bandwidth` accept one value or cyclic per-group values. `Legend` controls grouped legend visibility. `Parent` accepts a figure or axes handle in this build.",
            "MATLAB documents single/double observations and numeric controls. RunMat mode also accepts all eight fixed-width integer classes for X/Y, exact integer group labels, and supported numeric controls. Integer observations and floating-valued styling controls must be exactly representable when converted to binary64; integer grouping identity and bounded structural bin counts remain exact.",
            "The chart is composed on the host. RunMat mode admits resident GPU-array arguments through an owner-checked gather before chart construction; MATLAB mode rejects that extension before provider access. This is documented host materialization, not a GPU histogram kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "scatterhist-basic", title: "Create a scatter plot with marginal histograms",
            program: "h = scatterhist([0; 0.2; 0.8; 1], [0; 0.1; 0.9; 1], 'NBins', [2 2]); h",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(h),[3 1])); assert(strcmp(get(get(h(2),'Title'),'String'),'X Marginal')); assert(strcmp(get(get(h(3),'Title'),'String'),'Y Marginal'));" },
        },
        BuiltinExample { id: "scatterhist-group-kernel", title: "Group observations and show kernel marginals",
            program: "x = [0; 0.2; 0.8; 1]; y = [0; 0.1; 0.9; 1]; g = [\"a\"; \"b\"; \"a\"; \"b\"]; h = scatterhist(x, y, 'Group', g, 'Kernel', 'overlay', 'Legend', 'on'); h",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(h),[3 1])); assert(get(h(1),'LegendVisible')); assert(strcmp(get(get(h(2),'Title'),'String'),'X Marginal'));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[SOURCE, ADMISSION, CONTROLS, STYLES, LABELS, NUMERIC, RENDER, MATLAB],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ADMISSION, CONTROLS, STYLES, LABELS, NUMERIC, RENDER],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scatterhist grouped rendering, options, integer boundaries and axes", location: "crates/runmat-runtime/src/builtins/stats/summary/scatterhist/tests.rs" }],
        notes: &["The graphics compositor publishes host axes after input and option admission."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
