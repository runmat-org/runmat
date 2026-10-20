use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rowfilter"),
    slug: Some("rowfilter"),
    summary: "Create a row-filter selector for tabular and Parquet workflows.",
    description: "`rowfilter(source)` creates a host-side row-filter selector from supported table, timetable, Parquet metadata, datastore, or variable-selection inputs. The descriptor records the source in `Variables` and its optional second argument in `Predicate`.",
    keywords: &[
        "rowfilter",
        "table",
        "rows",
        "filter",
    ],
    related: &[
        "vartype",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The selector is a host metadata object intended for table, timetable, datastore, and Parquet row-filtering workflows.",
                "Direct native integer arrays and gpuArray values are not constructor roles and reject before gathering; integer-valued columns nested inside an admitted table or timetable remain exact opaque payloads.",
                "The optional second argument is retained as Predicate for table indexing; construction does not evaluate it.",
                "rowfilter() stores an empty 0-by-0 cell Variables and an empty string Predicate. A source-only form stores an empty Predicate; a second payload is retained without invocation. More than two inputs reject with RunMat:table:InvalidArgument before gathering.",
                "The constructor records metadata only: it neither evaluates predicates nor opens files. Table indexing supports the existing named predicate and logical-mask subset; Parquet row-filter pushdown and arbitrary predicate callbacks are not provided by this constructor.",
                "The table indexing example uses the supported @gt0 named predicate. Using this descriptor does not enable Parquet predicate pushdown or datastore iteration.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Data handling and outputs",
            paragraphs: &[
                "Filter descriptors are host metadata.",
                "The result is a scalar host descriptor. Accepted resident payloads are gathered without changing their numeric class, shape, complexity, or exact integer values. Numeric controls follow the restrictions listed above.",
                "Each call returns one descriptor. Calls with no retained output still validate their inputs; requesting more than one output is an error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "standalone-descriptor-contract",
            title: "Filter rows",
            program: "T = table([-1; 2; 3], 'VariableNames', {'A'});\nF = rowfilter({'A'}, '@gt0');\nR = T(F, :)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(F, 'rowfilter'));\nassert(strcmp(F.Predicate, '@gt0'));\nassert(isequal(R.A, [2; 3]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "vartype",
            target: BuiltinDocumentationLinkTarget::Builtin("vartype"),
        },
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/table/builtins/constructors.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/constructors.rs"),
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
