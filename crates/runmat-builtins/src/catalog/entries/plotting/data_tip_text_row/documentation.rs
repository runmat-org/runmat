use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dataTipTextRow"),
    slug: Some("dataTipTextRow"),
    summary: "Create a data-tip text row object for customizing plot data tips.",
    description: "`dataTipTextRow(label, value)` creates a MATLAB-compatible `matlab.graphics.datatip.DataTipTextRow` handle object with `Label`, `Value`, and `Format` properties. Use these rows with plot data-tip templates as RunMat's graphics template support expands.",
    keywords: &[
        "dataTipTextRow",
        "data tip",
        "datatip",
        "data cursor",
        "plotting",
        "graphics",
    ],
    related: &[
        "line",
        "plot",
        "scatter",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`dataTipTextRow(label, value)` creates a handle-class row object and stores the documented default `Format` value `auto`.",
                "`dataTipTextRow(label, value, format)` stores the supplied display format string.",
                "`label` and `format` accept character vectors, string scalars, and scalar string arrays.",
                "Documented text, scalar/vector, and function-handle Value sources are preserved without reinterpretation. Integer-vector support is classified as an evidence-based inference from the documented vector contract because the public page has no per-class data-type table.",
                "GPU-resident Value sources require the `data-tip-text-row-resident-value` extension and remain untouched. Host nonvector numeric sources independently require `data-tip-text-row-nonvector-value`.",
                "Returned rows support dot-property access and mutation for `Label`, `Value`, and `Format`; aliases refer to the same underlying handle object.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Properties",
            paragraphs: &[
                "`Label` is the displayed row label.",
                "`Value` is the data source, value vector, expression result, or property name used by the row.",
                "`Format` is an optional numeric, datetime, duration, or text display format.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Limitations",
            paragraphs: &[
                "DataTipTemplate attachment, source-length validation against chart data, and final numeric/datetime/duration rendering remain graphics-host integration work.",
                "The public documentation does not resolve per-class complex-vector or GPU Value support; RunMat treats resident values as an explicit extension.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`dataTipTextRow` constructs metadata. It stores the supplied value object as-is, including GPU-resident tensor handles, and does not gather numeric payloads.",
                "`dataTipTextRow` is metadata-only. It does not inspect, gather, or convert the supplied `Value` property, so GPU-resident values remain represented by their existing handles.",
                "Object construction terminates numeric fusion because the result is a graphics metadata object.",
                "Supplying a GPU-resident value preserves the handle in the returned object's `Value` property only when the resident-value extension is enabled.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create a row that reads from plotted Y data",
            program: "row = dataTipTextRow('Speed', 'YData', '%.2f');\nrow.Label",
            display_output: Some("ans =\n    \"Speed\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(row.Label, \"Speed\")); assert(strcmp(row.Value, \"YData\")); assert(strcmp(row.Format, \"%.2f\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Create a row from explicit values",
            program: "values = [10 20 30];\nrow = dataTipTextRow('Amplitude', values);\nrow.Format = '%.1f';",
            display_output: Some("% row is a DataTipTextRow handle object"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(row.Value, values)); assert(strcmp(row.Format, \"%.1f\")); alias = row; alias.Label = \"Magnitude\"; assert(strcmp(row.Label, \"Magnitude\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does dataTipTextRow render data tips by itself?",
            answer: "No. It creates the row object used by data-tip templates. Rendering and interactive data cursor behavior are handled by graphics hosts and future template integration.",
        },
        BuiltinDocumentationFaq {
            question: "Does it gather GPU data?",
            answer: "Not in strict compatibility mode. With the `data-tip-text-row-resident-value` extension enabled, RunMat preserves the supplied GPU handle as metadata without a host round trip.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "line",
            target: BuiltinDocumentationLinkTarget::Builtin("line")
        },
        BuiltinDocumentationLink {
            label: "plot",
            target: BuiltinDocumentationLinkTarget::Builtin("plot")
        },
        BuiltinDocumentationLink {
            label: "scatter",
            target: BuiltinDocumentationLinkTarget::Builtin("scatter")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/plotting/ops/data_tip_text_row.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/data_tip_text_row.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::plotting::data_tip_text_row::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "crates/runmat-vm/tests/plotting.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
