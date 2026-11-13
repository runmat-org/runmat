use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("textscatter3"),
    slug: Some("textscatter3"),
    summary: "Create a 3-D text scatter chart from x/y/z coordinates and text labels.",
    description: "`textscatter3(x,y,z,str)` creates a 3-D text scatter chart with labels from `str` at matching x/y/z coordinates. `textscatter3(xyz,str)` accepts an N-by-3 coordinate matrix. The returned handle is a `textscatter` chart object with data and display properties.",
    keywords: &["textscatter3", "3-D text scatter", "text analytics", "word embedding plot", "3-D text plot", "matlab textscatter3"],
    related: &["textscatter", "scatter3", "text", "get", "set"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`x`, `y`, `z`, and `str` must have equal element counts.",
                "`xyz` must be an N-by-3 matrix whose first, second, and third columns provide x, y, and z coordinates.",
                "Coordinates can use single, double, or any of the eight integer classes. `XData`, `YData`, and `ZData` retain their native class, shape, and exact values.",
                "`str` can be a string scalar, string array, character row, character matrix, or cell array of text scalars.",
                "The axes-target form `textscatter3(ax,...)` renders into the supplied axes handle.",
                "`ZData` is populated from the z coordinates and can be queried with `get`.",
                "`TextDensityPercentage`, `MaxTextLength`, `MarkerColor`, `MarkerSize`, `ColorData`, `Colors`, and `Visible` follow the same semantics as `textscatter`.",
                "`get` supports `Type`, `Parent`, `Children`, `TextData`, `XData`, `YData`, `ZData`, `TextDensityPercentage`, `MaxTextLength`, `MarkerColor`, `MarkerSize`, `ColorData`, `Colors`, `Visible`, `DisplayName`, and `Tag`.",
                "`set` supports `TextData`, `XData`, `YData`, `ZData`, `TextDensityPercentage`, `MaxTextLength`, `MarkerColor`, `MarkerSize`, `ColorData`, `Colors`, `Visible`, `Color`, `FontSize`, `FontWeight`, `FontAngle`, and `Interpreter`.",
                "Categorical `ColorData`, datasource properties, callbacks, app data, and interactive data-tip template objects are not implemented yet and raise explicit unsupported-property errors.",
                "Typed-integer numeric axes aliases and explicit gpuArray inputs are RunMat-mode extensions.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`textscatter3` is a host chart and annotation sink. Automatically resident inputs are gathered through their exact owning provider. Explicit gpuArray input is a RunMat-mode extension."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Three native coordinate arrays remain authoritative; floating renderer geometry is a derived cache. Creation gathers resident inputs asynchronously through their owning provider; explicit gpuArray is extension-gated before gathering. Complete finite scratch/state/render ownership and the related property-mutation pipeline remain separate obligations."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create a 3-D text scatter chart",
            program: "x = [1 2 3];\ny = [4 5 6];\nz = [7 8 9];\nwords = [\"one\" \"two\" \"three\"];\nts = textscatter3(x, y, z, words);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(ts, \"Type\"), \"textscatter\")); assert(isequal(get(ts, \"XData\"), x)); assert(isequal(get(ts, \"YData\"), y)); assert(isequal(get(ts, \"ZData\"), z)); assert(isequal(get(ts, \"TextData\"), words));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use an N-by-3 coordinate matrix",
            program: "xyz = [1 4 7; 2 5 8; 3 6 9];\nts = textscatter3(xyz, [\"one\" \"two\" \"three\"]);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "actualX = get(ts, \"XData\"); assert(isequal(actualX(:), xyz(:,1))); actualY = get(ts, \"YData\"); assert(isequal(actualY(:), xyz(:,2))); actualZ = get(ts, \"ZData\"); assert(isequal(actualZ(:), xyz(:,3)));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Read z coordinates from the chart",
            program: "ts = textscatter3([1 2], [3 4], [5 6], [\"a\" \"b\"]);\nget(ts, 'ZData')",
            display_output: Some("ans =\n     5     6"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(get(ts, \"ZData\"), [5 6])); assert(isequal(get(ts, \"TextData\"), [\"a\" \"b\"]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does textscatter3 return a different object type than textscatter?",
            answer: "The MATLAB chart type is `textscatter` for both 2-D and 3-D forms. RunMat follows that convention and distinguishes the 3-D form by storing nonempty `ZData`.",
        },
        BuiltinDocumentationFaq {
            question: "How are 3-D markers handled?",
            answer: "When markers are enabled, RunMat creates an internal scatter3 marker plot using the same coordinates and resolved marker colors.",
        },
        BuiltinDocumentationFaq {
            question: "Can textscatter3 preserve gpuArray residency?",
            answer: "The chart and its annotations are host graphics objects. Automatically resident inputs are gathered transparently through their owning provider. Explicit gpuArray inputs require RunMat mode.",
        },
        BuiltinDocumentationFaq {
            question: "Does `get(ts,'ZData')` preserve integer values?",
            answer: "Yes. All three coordinate properties use authoritative native storage, including `int64` and `uint64` values outside the exact double range. Renderer geometry is a separate derived cache.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "textscatter",
            target: BuiltinDocumentationLinkTarget::Builtin("textscatter"),
        },
        BuiltinDocumentationLink {
            label: "scatter3",
            target: BuiltinDocumentationLinkTarget::Builtin("scatter3"),
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
