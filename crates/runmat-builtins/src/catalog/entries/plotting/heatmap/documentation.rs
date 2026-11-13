use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Matrix and table forms",
        paragraphs: &[
            "`heatmap(CData)` and `heatmap(CData,Name,Value,...)` use default row and column labels. `heatmap(XValues,YValues,CData)` accepts text, string-array, cell-array, and numeric labels for the displayed columns and rows.",
            "The returned heatmap handle participates in RunMat's plotting property system. All eight integer matrix ColorData classes remain exact in public chart ColorData; separate exact range-relative normalization supplies the floating renderer without conflating adjacent wide integers.",
            "ColorLimits, FontSize, ColorbarVisible, and GridVisible accept documented integer values with exact structural validation. Public ColorLimits remain in the source ColorData domain, not the renderer's normalized domain.",
            "`heatmap(T,XVariable,YVariable)` groups rows of a table or timetable by two selected discrete single-column variables. Variable selectors use the same names, numeric positions, and one-hot logical masks as table indexing. The derived ColorData and labels are retained by the existing heatmap scene and figure replay path.",
            "Without ColorVariable, table ColorMethod defaults to count. With ColorVariable, it defaults to mean. Explicit count ignores ColorVariable values and counts every observed axis pair, including rows with a missing color observation.",
            "ColorMethod accepts count, mean, median, sum, min, max, and none. None requires unique observed axis pairs. Missing axis-pair combinations produce zero for count/sum and NaN otherwise. Numeric NaN color observations are omitted by summary methods. None retains Single and integer ColorData in the source class when every axis pair is present; incomplete integer grids require a floating NaN result and reject values that cannot cross that boundary exactly.",
            "RunMat currently excludes observations with a missing XVariable or YVariable from table heatmap grouping; a literal text category named 'missing' remains a normal category. MATLAB parity of this missing-axis row policy requires separate qualification.",
            "Axis values are ordered by their typed table grouping identity. Categorical values retain declared category order; other supported discrete values are sorted by value, not by formatted display label.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and current limits",
        paragraphs: &[
            "Heatmap is a client-side plotting sink and fusion boundary. Matrix GPU CData remains a gated RunMat extension and is downloaded non-destructively after admission. Resident columns inside a table are rejected before scene mutation until a table-column provider path exists.",
            "Saved figures retain the aggregated color values and axis labels. The table is not retained as a mutable SourceTable: changing XVariable, YVariable, ColorVariable, or ColorMethod after construction is not supported. With ColorMethod='none', an incomplete grid containing wide integers that cannot be represented exactly as double raises UnsupportedColorData instead of rounding those values.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Create a labeled heatmap chart",
        program: "cdata = [45 60 32; 43 54 76; 32 94 68; 23 95 58];\nxvalues = {'Small','Medium','Large'};\nyvalues = {'Green','Red','Blue','Gray'};\nh = heatmap(xvalues,yvalues,cdata);\nh.Title = 'T-Shirt Orders';\nh.XLabel = 'Sizes';\nh.YLabel = 'Colors';",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(isequal(get(h,'ColorData'),cdata)); assert(strcmp(get(h,'Type'),'heatmap'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Create a heatmap from matrix data",
        program: "A = magic(5);\nh = heatmap(A);\nh.Colormap = 'hot';\nh.ColorbarVisible = true;",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(isequal(get(h,'ColorData'),A)); assert(get(h,'ColorbarVisible'));",
        },
    },
    BuiltinExample {
        id: "table-mean",
        title: "Aggregate repeated table observations",
        program: "Hour = [8;8;9]; Fixture = [\"A\";\"A\";\"B\"]; Force = [2;4;10]; T = table(Hour,Fixture,Force); h = heatmap(T,'Hour','Fixture','ColorVariable','Force','ColorMethod','mean');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "C = get(h,'ColorData'); assert(isequaln(C,[3 NaN; NaN 10]));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("heatmap"),
    slug: Some("heatmap"),
    summary: "Create a heatmap chart from matrix data or aggregated table variables.",
    description: "`heatmap` creates a flat labeled grid whose authoritative ColorData is retained with its graphics handle. The table form groups observations before using the same scene and renderer as the matrix form.",
    keywords: &["heatmap", "HeatmapChart", "matrix visualization", "plotting", "colormap", "table aggregation"],
    related: &["imagesc", "table", "pivot", "colormap"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does heatmap support table inputs?",
        answer: "Yes. Use `heatmap(T,xvar,yvar)` for counts or add `ColorVariable` for a numeric/logical aggregation. The derived chart is supported; SourceTable and mutable table-variable properties remain an open compatibility gap.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "imagesc", target: BuiltinDocumentationLinkTarget::Builtin("imagesc") },
        BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
        BuiltinDocumentationLink { label: "colorbar", target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/heatmap.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Table aggregation and scene construction",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/heatmap.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Typed table grouping and retained ColorData",
            location: "builtins::plotting::heatmap::tests",
        }],
        notes: &["Original legacy JSON source programs are conserved here. Browser scene rendering, figure replay, native and WASM execution, and SourceTable/property parity require independent qualification. No generated JSON has been deleted or hand-edited."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
