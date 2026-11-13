use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("parquetDatastore"),
    slug: Some("parquetDatastore"),
    summary: "Create a parquet datastore descriptor object.",
    description: "`parquetDatastore(files, ...)` creates a validated descriptor object for parquet table workflows.",
    keywords: &[
        "parquetDatastore",
        "datastore",
        "parquet",
        "table",
    ],
    related: &[
        "arrayDatastore",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The constructor preserves file specifications and accepts supported `ReadSize` and `BlockSize` controls as positive doubles; `ReadSize` also accepts `\"file\"` and `\"rowgroup\"`.",
                "Typed-integer constructor arguments are rejected before gathering because the documented numeric constructor controls use double rather than native integer classes.",
                "Unknown options and invalid option pairs raise explicit errors instead of being ignored.",
                "Parquet decoding and iterator-style operations such as `read`, `hasdata`, and `reset` require the generic datastore dispatch layer and are not provided by this constructor builtin.",
                "Defaults are ReadSize='rowgroup' and BlockSize=128000000. Names are case-insensitive, repeated options use the last value, and accepted ReadSize text retains its input case.",
                "File specifications are retained, not opened, expanded, or validated for existence. This constructor therefore needs no filesystem capability or Parquet decoder. It does not implement predicate pushdown.",
                "Missing files arguments, invalid pairs, unsupported names, and invalid controls use RunMat:table:InvalidArgument. Some invalid numeric controls currently report an arrayDatastore-prefixed message.",
                "The shared pre-gather integer guard covers direct native integer storage, including sparse/complex representations, and recursive cell, struct, object, array, and closure-capture payloads. Rejection occurs before reading resident inputs.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Data handling and outputs",
            paragraphs: &[
                "Datastore descriptors are host metadata.",
                "The result is a scalar host descriptor. Accepted resident payloads are gathered without changing their numeric class, shape, complexity, or exact integer values. Numeric controls follow the restrictions listed above.",
                "Each call returns one descriptor. Calls with no retained output still validate their inputs; requesting more than one output is an error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "standalone-descriptor-contract",
            title: "Create parquet datastore descriptor",
            program: "ds = parquetDatastore('data.parquet')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(ds, 'parquetDatastore'));\nassert(strcmp(ds.Files, 'data.parquet'));\nassert(strcmp(ds.ReadSize, 'rowgroup'));\nassert(ds.BlockSize == 128000000);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "arrayDatastore",
            target: BuiltinDocumentationLinkTarget::Builtin("arrayDatastore"),
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
        ],
        notes: &[
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
