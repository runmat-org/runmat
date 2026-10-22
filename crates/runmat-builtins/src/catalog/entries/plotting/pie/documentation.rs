use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("pie"),

    slug: Some("pie"),

    summary: "Create pie charts in MATLAB and RunMat.",

    description: concat!(
        "`pie` creates part-to-whole charts from numeric slice values. Handle behavior, ",
        "explode controls, and labeling follow MATLAB semantics.",
    ),

    keywords: &[
        "pie",
        "pie chart",
        "exploded pie chart",
        "slice labels",
        "matlab pie",
    ],

    related: &["bar", "histogram", "legend"],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",

            paragraphs: &[
                concat!(
                    "Typed-integer slice values are a RunMat-mode extension and cross an ",
                    "exactness-checked floating renderer boundary; typed explode controls are ",
                    "interpreted structurally.",
                ),
                "Slice values determine the relative angular extent of each wedge.",
                "Explode-style workflows and slice labels are part of the pie rendering path.",
                concat!(
                    "The returned value is a pie-handle object participating in the same broader ",
                    "plotting handle model used across the plotting stack.",
                ),
                concat!(
    "Pie labels and overlay semantics survive replay/export because they are part of ",

    "the shared event/scene path.",

),
                concat!(
                    "Pie charts are especially useful as the part-to-whole complement to bar and ",
                    "histogram-style quantitative plots.",
                ),
            ],
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
    "Accepted forms are nonnegative finite slice values with optional explode vector ",

    "and labels/format, plus an optional full leading axes target. Slice integer ",

    "data uses its existing exactness-checked RunMat extension before floating ",

    "geometry. Explode controls are finite same-length numeric values interpreted by ",

    "nonzero, not arbitrary plot labels; an attempted provider readback failure now ",

    "propagates rather than being swallowed as a failed label probe.",

),
                concat!(
                    "Labels accept cell text, string arrays, text scalars or format strings; ",
                    "explicit labels must match slice count. Shared PieChart owns relative slice ",
                    "geometry, labels and default percentage presentation. The current public pie ",
                    "getter exposes Type/Parent/Children/DisplayName, not per-slice vertices or ",
                    "explode arrays: doc Figure witnesses do not pretend otherwise, and retained ",
                    "numerical/geometry Runtime fixtures are a distinct gate. Full MATLAB ",
                    "patch-array outputs and properties are not implemented by this scalar ",
                    "pie-handle owner.",
                ),
            ],
        },
    ],

    examples: &[
        BuiltinExample {
            id: "retained-1",

            title: "Create a basic pie chart",

            program: "pie([2 3 5]);",

            display_output: None,

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "witness=findobj(gca,'Type','pie'); assert(numel(witness)>=1); ",
                    "assert(numel(witness)==1); assert(get(witness,'Parent')==gca); ",
                ),
            },
        },
        BuiltinExample {
            id: "retained-2",

            title: "Create an exploded pie chart",

            program: "values = [4 2 3 1];\nexplode = [1 0 1 0];\npie(values, explode);",

            display_output: None,

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "witness=findobj(gca,'Type','pie'); assert(numel(witness)>=1); ",
                    "assert(numel(witness)==1); assert(get(witness,'Parent')==gca); ",
                ),
            },
        },
        BuiltinExample {
            id: "retained-3",

            title: "Use labels for slice names",

            program:
                "values = [5 3 2];\nlabels = {'compute', 'io', 'render'};\npie(values, labels);",

            display_output: None,

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "witness=findobj(gca,'Type','pie'); assert(numel(witness)>=1); ",
                    "assert(numel(witness)==1); assert(get(witness,'Parent')==gca); ",
                ),
            },
        },
        BuiltinExample {
            id: "retained-4",

            title: "Budget allocation",

            program: concat!(
                "values = [35 25 20 12 8];\nlabels = {'Engineering', 'Marketing', 'Operations', ",
                "'R&D', 'Admin'};\nexplode = [1 0 0 0 0];\n\npie(values, explode);\ntitle('Annual ",
                "Budget Allocation');\nlegend(labels);",
            ),

            display_output: None,

            compatibility: BuiltinExampleCompatibility::Matlab,

            harness: BuiltinExampleHarness::BrowserGraphics,

            fixture: BuiltinExampleFixture::None,

            requirements: BuiltinExampleRequirements::NONE,

            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,

                assertions: concat!(
                    "witness=findobj(gca,'Type','pie'); assert(numel(witness)>=1); ",
                    "assert(numel(witness)==1); assert(get(witness,'Parent')==gca); ",
                    "assert(isequal(values,[35 25 20 12 8]));",
                ),
            },
        },
    ],

    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "How do I explode specific slices out of the pie?",

            answer: concat!(
    "Pass an explode vector the same length as your values. A `1` pulls that slice ",

    "outward; a `0` keeps it flush.\n\n```matlab\npie([4 2 3 1], [0 1 0 0]);\n```\n\nThis ",

    "explodes only the second slice. You can explode multiple slices by setting more ",

    "entries to `1`.",

),
        },
        BuiltinDocumentationFaq {
            question: "How do I add percentage labels to slices?",

            answer: concat!(
                "By default, `pie` displays percentage labels on each slice. To use custom ",
                "labels instead, pass a cell array of strings as the second ",
                "argument.\n\n```matlab\npie([5 3 2], {'compute 50%', 'io 30%', 'render ",
                "20%'});\n```\n\nIf you want no labels at all, pass empty strings for each slice.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "When should I use a pie chart vs a bar chart?",

            answer: concat!(
                "Pie charts work well when you have a small number of slices (2–5) and the point ",
                "is to show each slice's share of a whole. Once you go beyond ~6 slices or need ",
                "to compare absolute values across categories, `bar` is almost always ",
                "clearer—humans are better at comparing lengths than angles. If two slices are ",
                "close in size and the difference matters, use `bar`.",
            ),
        },
    ],

    links: &[
        BuiltinDocumentationLink {
            label: "bar",

            target: BuiltinDocumentationLinkTarget::Builtin("bar"),
        },
        BuiltinDocumentationLink {
            label: "histogram",

            target: BuiltinDocumentationLinkTarget::Builtin("histogram"),
        },
        BuiltinDocumentationLink {
            label: "legend",

            target: BuiltinDocumentationLinkTarget::Builtin("legend"),
        },
        BuiltinDocumentationLink {
            label: "Choosing the right plot type",

            target: BuiltinDocumentationLinkTarget::Documentation(
                "/docs/plotting/choosing-the-right-plot-type",
            ),
        },
        BuiltinDocumentationLink {
            label: "Styling plots and axes",

            target: BuiltinDocumentationLinkTarget::Documentation(
                "/docs/plotting/styling-plots-and-axes",
            ),
        },
        BuiltinDocumentationLink {
            label: "Plot replay and export",

            target: BuiltinDocumentationLinkTarget::Documentation(
                "/docs/plotting/plot-replay-and-export",
            ),
        },
        BuiltinDocumentationLink {
            label: "Complete plotting guide",

            target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide"),
        },
    ],

    media: &[BuiltinDocumentationMedia {
        kind: BuiltinDocumentationMediaKind::Image,

        url: "https://web.runmatstatic.com/builtin-image/runmat-matlab-plot-apie-budget.webp",

        alt: "Budget allocation",
    }],

    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/pie.rs`",

            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",
                "s/plotting/ops/pie.rs",
            )),
        }],

        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,

                label: "unit",

                location: "builtins::plotting::pie::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,

                label: "integration",

                location: "runmat-plot/tests/renderer_tests.rs",
            },
        ],

        notes: &[],
    },

    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
