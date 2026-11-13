use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("histogram2"),

    slug: Some("histogram2"),

    summary: "Create two-dimensional histogram chart objects in MATLAB and RunMat.",

    description: concat!(
        "`histogram2(X,Y)` bins paired observations with MATLAB-compatible `histcounts2` ",
        "semantics and creates a handle-backed 2-D histogram chart. RunMat renders the ",
        "chart as a colormapped surface: `DisplayStyle` `\"bar3\"` uses bin counts as ",
        "surface height, while `\"tile\"` renders a flattened heatmap-style view.",
    ),

    keywords: &[
        "histogram2",
        "histcounts2",
        "2d histogram",
        "density plot",
        "bivariate histogram",
        "plotting",
    ],

    related: &["histcounts2", "histogram", "get", "set"],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",

            paragraphs: &[
                concat!(
    "`histogram2(X,Y)` requires `X` and `Y` to contain the same number of elements ",

    "and ignores pairs containing `NaN`.",

),
                concat!(
    "Positional bin arguments and name/value bin controls are forwarded through the ",

    "same implementation as `histcounts2`, including `NumBins`, explicit ",

    "`XBinEdges`/`YBinEdges`, `BinWidth`, `BinLimits`, `BinMethod`, axis-specific ",

    "bin controls, and normalization modes.",

),
                concat!(
                    "Returned handles have type `histogram2` and expose `Values`, `BinCounts`, ",
                    "`XBinEdges`, `YBinEdges`, `NumBins`, `Normalization`, `DisplayStyle`, ",
                    "`ShowEmptyBins`, `FaceAlpha`, and `DisplayName` through `get`.",
                ),
                concat!(
                    "`set` supports updating `Normalization`, `DisplayStyle`, `ShowEmptyBins`, ",
                    "`FaceAlpha`, and `DisplayName`, rebuilding the rendered surface and handle ",
                    "state.",
                ),
                concat!(
    "Common styling name/value pairs `FaceColor`, `EdgeColor`, `EdgeAlpha`, and ",

    "`LineStyle` are accepted for MATLAB script compatibility, but RunMat currently ",

    "maps the chart through a colormapped surface backend rather than per-bin bar ",

    "primitives.",

),
            ],
        },
        BuiltinDocumentationSection {
            heading: "Rendering and residency",

            paragraphs: &[concat!(
                "`histogram2` is a plotting sink built on `histcounts2`; current builds ",
                "materialize the bin grid before rendering.",
            )],
        },
        BuiltinDocumentationSection {
            heading: "Declared source precision and class coverage",

            paragraphs: &["single", "double"],
        },
        BuiltinDocumentationSection {
            heading: "Coordinate broadcasting",

            paragraphs: &["none"],
        },
        BuiltinDocumentationSection {
            heading: "Runtime contract and boundaries",

            paragraphs: &[
                concat!(
    "Catalog default bindings retain the actual RuntimeContext and original rooted ",

    "BuiltinArguments across parsing, provider readback, delegated calls and ",

    "completion. Value-selected axes, properties and output counts remain Runtime ",

    "decisions. Constructor results are host double graphics handles, not numerical ",

    "GPU arrays; hist instead returns double counts/centres, plotyy up to three ",

    "distinct handle outputs, and plotmatrix up to five.",

),
                concat!(
    "Rendering is a fusion boundary and requires the installed graphics host/device. ",

    "A direct compatible renderer route is not an elementwise or reduction provider ",

    "kernel. Resident buffers retain their owning provider and existing ",

    "export/error/completion protocol; an export, readback, allocation, cancellation ",

    "or emission failure is an error, not permission to replay an attempted ",

    "operation. Source data and renderer caches have distinct lifetimes. No new ",

    "distributed chart implementation is declared.",

),
                concat!(
    "The existing canonical CoordinateWork allocation policy refuses Managed calls ",

    "before unowned chart scratch, metadata, readback or scene mutation. It checks ",

    "captured cancellation generation and policy before publication and on ",

    "completion; ingress failures are Never replay. This is an explicit missing ",

    "finite owner, not an allocation estimate, algorithm completion or L25 closure. ",

    "Provider-private allocations, renderer lifetime accounting, synchronous long ",

    "geometry/copy cooperation, measured throughput and physical presentation remain ",

    "independent qualification gates.",

),
                concat!(
    "Native Single/Double/integer source ownership is distinguished from explicit ",

    "binary64/f32 graphics conversion. The renderer does not promise exact ",

    "wide-integer pixel positions. Logical, complex, sparse, scalar, vector, matrix, ",

    "empty and higher-rank forms follow the identity-specific parser rather than a ",

    "universal implicit cast. Sparse/complex rejection is not an integer ",

    "not-applicable audit. Full MATLAB properties, browser/native parity and ",

    "empty-shape conformance are not inferred from migration or historical test ",

    "references.",

),
            ],
        },
        BuiltinDocumentationSection {
            heading: "Supported representations and remaining forms",

            paragraphs: &[
                concat!(
    "Paired X/Y samples must have compatible element counts. Existing histcounts2 ",

    "owns bin controls and normalization; retained chart Data is constructed once ",

    "before chart evaluation. NumBins and axis-specific BinEdges/Width/Limits/Method ",

    "options preserve their existing grammar. DisplayStyle bar3 creates a raised ",

    "colormapped surface, tile a flattened image-style surface; ShowEmptyBins, ",

    "FaceAlpha and DisplayName are actual controls.",

),
                concat!(
    "FaceColor/EdgeColor/EdgeAlpha/LineStyle are accepted compatibility tokens but ",

    "the current colormapped surface backend does not implement independent per-bin ",

    "bar edges; they must not be advertised as full rendered property support. ",

    "Values and raw BinCounts/XBinEdges/YBinEdges/NumBins remain separately ",

    "queryable. The histogram owner preserves typed source diagnostics and rejects ",

    "output over-request before effects. Direct finite/native provider binning is ",

    "not added here.",

),
            ],
        },
    ],

    examples: &[
        BuiltinExample {
            id: "retained-1",

            title: "Create a 2-D histogram with two bins per axis",

            program: "h = histogram2([0; 0.2; 0.8; 1], [0; 0.1; 0.9; 1], [2 2]);\nget(h, 'Values')",

            display_output: Some("ans =\n     2     0\n     0     2"),

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "observed=get(h,'BinCounts'); expected=[2 0;0 2]; ",
                    "assert(isequal(observed(:),expected(:))); assert(isequal(get(h,'NumBins'),[2 ",
                    "2]));",
                ),
            },
        },
        BuiltinExample {
            id: "retained-2",

            title: "Use probability normalization and tile display",

            program: concat!(
                "% Deterministic source variables for this standalone ",
                "example.\nx=linspace(0,1,11); y=linspace(-1,1,11);\nh = histogram2(x, y, ",
                "'NumBins', [20 10], 'Normalization', 'probability', 'DisplayStyle', ",
                "'tile');\nget(h, 'Normalization')",
            ),

            display_output: Some("ans =\n    'probability'"),

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "assert(strcmp(get(h,'Normalization'),'probability')); ",
                    "assert(strcmp(get(h,'DisplayStyle'),'tile')); ",
                    "assert(isequal(get(h,'NumBins'),[20 10])); observed=get(h,'Values'); ",
                    "assert(abs(sum(observed(:))-1)<1e-12);",
                ),
            },
        },
        BuiltinExample {
            id: "retained-3",

            title: "Inspect and update chart properties",

            program: concat!(
                "% Deterministic source variables for this standalone ",
                "example.\nx=linspace(0,1,11); y=linspace(-1,1,11);\nh = histogram2(x, y, ",
                "'XBinEdges', 0:0.1:1, 'YBinEdges', -1:0.2:1);\nset(h, 'DisplayStyle', 'tile', ",
                "'ShowEmptyBins', 'off');\nget(h, 'NumBins')",
            ),

            display_output: Some("ans = [10 10]"),

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "assert(strcmp(get(h,'DisplayStyle'),'tile')); ",
                    "assert(isequal(get(h,'NumBins'),[10 10])); observed=get(h,'BinCounts'); ",
                    "assert(sum(observed(:))==11);",
                ),
            },
        },
    ],

    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `histogram2` related to `histcounts2`?",

            answer: concat!(
                "`histogram2` uses the same binning and normalization logic as `histcounts2`, ",
                "then wraps the resulting count matrix in a plotting object.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Which display styles are supported?",

            answer: concat!(
                "`'bar3'` and `'tile'` are supported. `bar3` is rendered as a raised colormapped ",
                "count surface; `tile` is rendered as a flattened colormapped surface with ",
                "image-mode camera behavior.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Can `histogram2` plot GPU arrays?",

            answer: concat!(
                "Yes, but today it follows the underlying `histcounts2` gather-immediate path ",
                "and renders from the materialized bin grid. Future provider kernels can ",
                "accelerate the binning stage without changing the plotting API.",
            ),
        },
    ],

    links: &[
        BuiltinDocumentationLink {
            label: "histcounts2",

            target: BuiltinDocumentationLinkTarget::Builtin("histcounts2"),
        },
        BuiltinDocumentationLink {
            label: "histogram",

            target: BuiltinDocumentationLinkTarget::Builtin("histogram"),
        },
        BuiltinDocumentationLink {
            label: "get",

            target: BuiltinDocumentationLinkTarget::Builtin("get"),
        },
        BuiltinDocumentationLink {
            label: "set",

            target: BuiltinDocumentationLinkTarget::Builtin("set"),
        },
    ],

    media: &[],

    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/histogram2.rs`",

            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",
                "s/plotting/ops/histogram2.rs",
            )),
        }],

        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,

                label: "unit",

                location: "builtins::plotting::histogram2::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,

                label: "vm",

                location: "crates/runmat-vm/tests/plotting.rs",
            },
        ],

        notes: &[],
    },

    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
