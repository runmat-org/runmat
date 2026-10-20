use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("uitable"),
    slug: Some("uitable"),
    summary: "Create a table UI component descriptor.",
    description: "`uitable('Data', A)` creates a descriptor object with `Data`, `ColumnName`, and `RowName` properties.",
    keywords: &[
        "uitable",
        "ui",
        "table",
        "Data",
    ],
    related: &[
        "table",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The first positional argument is used as data when no `\"Data\"` option is supplied.",
                "The `Data` property accepts every numeric type. All eight integer classes retain their exact class, shape, and values in the returned descriptor object.",
                "Automatically resident data may gather transparently for this host UI boundary. Explicit `gpuArray` data is a compatibility-gated RunMat extension.",
                "uitable() creates empty cell Data and empty 0-by-1 string ColumnName and RowName arrays. The first matching Data option at even-position name/value pairs wins. Otherwise the first argument becomes Data.",
                "Only Data is implemented as an input property. Other input properties are currently not applied or validated; ColumnName and RowName always start empty. This metadata API is not the full MATLAB UI component API.",
                "No window, widget, parent attachment, renderer, event loop, or callback invocation is created. Displaying an interactive UI is not supported.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Data handling and outputs",
            paragraphs: &[
                "UI descriptors are host metadata. Automatically resident data gathers through its owning provider; explicitly GPU-resident Data is available only as a RunMat extension.",
                "The result is a scalar host descriptor. Accepted resident payloads are gathered without changing their numeric class, shape, complexity, or exact integer values. Numeric controls follow the restrictions listed above.",
                "Each call returns one descriptor. Calls with no retained output still validate their inputs; requesting more than one output is an error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "standalone-descriptor-contract",
            title: "Create UI table descriptor",
            program: "u = uitable('Data', [1 2])",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(u, 'uitable'));\nassert(isequal(u.Data, [1 2]));\nassert(isempty(u.ColumnName));\nassert(isempty(u.RowName));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "table",
            target: BuiltinDocumentationLinkTarget::Builtin("table"),
        },
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/table/mod.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/mod.rs"),
            },
            BuiltinDocumentationLink {
                label: "Descriptor constructor and admission",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/constructors.rs"),
            },
            BuiltinDocumentationLink {
                label: "Table selector consumers",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/object/selectors.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Descriptor semantics, integers and validation",
                location: "crates/runmat-runtime/src/builtins/table/tests.rs",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Typed descriptor inference",
                location: "crates/runmat-builtins/src/catalog/entries/table/descriptor_objects/tests.rs",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Tabular indexing integration",
                location: "crates/runmat-vm/tests/tables_timetables.rs",
            },
        ],
        notes: &[
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
