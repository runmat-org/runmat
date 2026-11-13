use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("textscatter"),
    slug: Some("textscatter"),
    summary: "Create a 2-D text scatter chart from x/y coordinates and text labels.",
    description: "`textscatter(x,y,str)` creates a text scatter chart with labels from `str` at matching x/y coordinates. `textscatter(xy,str)` accepts an N-by-2 coordinate matrix. The returned handle provides chart data and display properties.",
    keywords: &["textscatter", "text scatter", "text analytics", "word embedding plot", "2-D text plot", "matlab textscatter"],
    related: &["textscatter3", "scatter", "text", "get", "set"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`x`, `y`, and `str` must have equal element counts.",
                "`xy` must be an N-by-2 matrix whose first and second columns provide x and y coordinates.",
                "Coordinates can use single, double, or any of the eight integer classes. `XData` and `YData` retain their native class, shape, and exact values.",
                "`str` can be a string scalar, string array, character row, character matrix, or cell array of text scalars.",
                "The axes-target form `textscatter(ax,...)` renders into the supplied axes handle.",
                "`TextDensityPercentage` controls how many labels are visible; `0` hides all text labels and `100` shows all labels and suppresses marker creation.",
                "`MaxTextLength` truncates long labels and adds `...` at the truncation point.",
                "`MarkerColor` accepts `auto`, `none`, or an RGB color. `ColorData` accepts an RGB triplet or N-by-3 RGB matrix for text colors.",
                "`get` supports `Type`, `Parent`, `Children`, `TextData`, `XData`, `YData`, `ZData`, `TextDensityPercentage`, `MaxTextLength`, `MarkerColor`, `MarkerSize`, `ColorData`, `Colors`, `Visible`, `DisplayName`, and `Tag`; 2-D charts report empty `ZData`.",
                "`set` supports `TextData`, `XData`, `YData`, `TextDensityPercentage`, `MaxTextLength`, `MarkerColor`, `MarkerSize`, `ColorData`, `Colors`, `Visible`, `Color`, `FontSize`, `FontWeight`, `FontAngle`, and `Interpreter`. `ZData` mutation is reserved for `textscatter3` charts.",
                "Categorical `ColorData`, datasource properties, callbacks, app data, and interactive data-tip template objects are not implemented yet and raise explicit unsupported-property errors.",
                "Typed-integer numeric axes aliases and explicit gpuArray inputs are RunMat-mode extensions.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`textscatter` is a host chart and annotation sink. Automatically resident inputs are gathered through their exact owning provider. Explicit gpuArray input is a RunMat-mode extension."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Native coordinate data remains authoritative; floating renderer geometry is a derived cache. Creation gathers resident inputs asynchronously through their owning provider; explicit gpuArray is extension-gated before gathering. The chart has no provider kernel. Complete finite scratch/state/render ownership and the related property-mutation pipeline remain separate obligations."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create a 2-D text scatter chart",
            program: "x = [1 2 3];\ny = [4 5 6];\nwords = [\"one\" \"two\" \"three\"];\nts = textscatter(x, y, words);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(ts, \"Type\"), \"textscatter\")); assert(isequal(get(ts, \"XData\"), x)); assert(isequal(get(ts, \"YData\"), y)); assert(isequal(get(ts, \"TextData\"), words)); assert(isempty(get(ts, \"ZData\")));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use an N-by-2 coordinate matrix",
            program: "xy = [1 4; 2 5; 3 6];\nts = textscatter(xy, [\"one\" \"two\" \"three\"]);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "actualX = get(ts, \"XData\"); assert(isequal(actualX(:), xy(:,1))); actualY = get(ts, \"YData\"); assert(isequal(actualY(:), xy(:,2))); assert(isequal(get(ts, \"TextData\"), [\"one\" \"two\" \"three\"]));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Limit label length and color each label",
            program: "xy = [1 1; 2 2; 3 3];\ncolors = [1 0 0; 0 1 0; 0 0 1];\nts = textscatter(xy, [\"alphabet\" \"betatron\" \"gamma\"], 'MaxTextLength', 5, 'ColorData', colors);\nget(ts, 'TextData')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(get(ts, \"TextData\"), [\"alphabet\" \"betatron\" \"gamma\"])); assert(get(ts, \"MaxTextLength\") == 5); assert(isequal(get(ts, \"ColorData\"), colors));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does textscatter return individual text handles?",
            answer: "No. RunMat returns a first-class `textscatter` chart handle. The chart owns the text annotations and marker plot internally, so chart data and display properties can be queried and updated through `get` and `set`.",
        },
        BuiltinDocumentationFaq {
            question: "How are markers handled?",
            answer: "When `TextDensityPercentage` is below 100 and `MarkerColor` is not `none`, RunMat creates an internal scatter marker plot. At 100 percent density, markers are suppressed to match MATLAB behavior.",
        },
        BuiltinDocumentationFaq {
            question: "Can textscatter preserve gpuArray residency?",
            answer: "The chart and its annotations are host graphics objects. Automatically resident inputs are gathered transparently through their owning provider. Explicit gpuArray inputs require RunMat mode.",
        },
        BuiltinDocumentationFaq {
            question: "Does `get(ts,'XData')` preserve integer values?",
            answer: "Yes. Coordinate properties use authoritative native storage, including `int64` and `uint64` values outside the exact double range. Renderer geometry is a separate derived cache.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "textscatter3",
            target: BuiltinDocumentationLinkTarget::Builtin("textscatter3"),
        },
        BuiltinDocumentationLink {
            label: "scatter",
            target: BuiltinDocumentationLinkTarget::Builtin("scatter"),
        },
        BuiltinDocumentationLink {
            label: "text",
            target: BuiltinDocumentationLinkTarget::Builtin("text"),
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
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/textscatter.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/textscatter.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::plotting::textscatter::tests",
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
