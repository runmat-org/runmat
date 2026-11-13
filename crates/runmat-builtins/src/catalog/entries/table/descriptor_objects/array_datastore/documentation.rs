use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("arrayDatastore"),
    slug: Some("arrayDatastore"),
    summary: "Create a datastore for an in-memory array.",
    description: "`arrayDatastore(A)` retains an in-memory array together with `ReadSize`, `IterationDimension`, and `OutputType` properties. Integer A remains in its exact native class.",
    keywords: &[
        "arrayDatastore",
        "datastore",
        "array",
        "ReadSize",
        "IterationDimension",
        "OutputType",
    ],
    related: &[
        "parquetDatastore",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`ReadSize` and `IterationDimension` must be positive integer-valued doubles; typed-integer controls reject. `OutputType` is `\"cell\"` or `\"same\"`, and `\"same\"` requires iteration dimension 1.",
                "Interactive resident input is a mode-gated RunMat extension. It gathers exact typed storage before host object construction and rejects in MATLAB-compatible mode.",
                "RunMat currently represents this as a value descriptor whose `Data` property exposes the retained payload. Stateful iterator operations such as `read`, `readall`, `preview`, `hasdata`, and `reset` require a datastore handle/dispatch layer and are not yet provided.",
                "Defaults are ReadSize=1, IterationDimension=1, OutputType='cell'. Names and OutputType are case-insensitive; repeated supported options use the last value. Empty and N-D data retain their input shape; the constructor does not enforce a nonempty matrix.",
                "The constructor requires A, validates option pairs and supported names, and rejects invalid double controls or the 'same'/dimension mismatch with RunMat:table:InvalidArgument. Resident inputs are read before options are validated, so a device read error may be reported before an invalid option.",
                "This is a value descriptor, not a resumable iterator. Constructing it does not read files or invoke callbacks.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Data handling and outputs",
            paragraphs: &[
                "Interactive resident input is a mode-gated RunMat extension that gathers before host datastore construction.",
                "The result is a scalar host descriptor. Accepted resident payloads are gathered without changing their numeric class, shape, complexity, or exact integer values. Numeric controls follow the restrictions listed above.",
                "Each call returns one descriptor. Calls with no retained output still validate their inputs; requesting more than one output is an error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "standalone-descriptor-contract",
            title: "Retain typed data with explicit read properties",
            program: "ds = arrayDatastore(int16([1;2]), 'ReadSize', 2, 'OutputType', 'same')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(ds, 'arrayDatastore'));\nassert(isa(ds.Data, 'int16'));\nassert(isequal(ds.Data, int16([1;2])));\nassert(ds.ReadSize == 2);\nassert(ds.IterationDimension == 1);\nassert(strcmp(ds.OutputType, 'same'));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "parquetDatastore",
            target: BuiltinDocumentationLinkTarget::Builtin("parquetDatastore"),
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
