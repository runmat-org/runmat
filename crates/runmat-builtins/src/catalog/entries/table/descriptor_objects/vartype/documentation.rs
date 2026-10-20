use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("vartype"),
    slug: Some("vartype"),
    summary: "Create a table variable type selector.",
    description: "`vartype(type)` creates a selector with a `Type` property for table variable type selection workflows.",
    keywords: &[
        "vartype",
        "table",
        "variable type",
        "selector",
    ],
    related: &[
        "rowfilter",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`type` must be a string scalar or character row vector.",
                "The selector can be used as a table or timetable variable subscript.",
                "Supported categories include numeric, logical, string, cell, categorical, datetime, duration, integer, and all.",
                "The text selector `\"integer\"` selects integer variables; it is not an integer-valued argument. Numeric and provider-resident values are rejected before provider access.",
                "The constructor preserves selector spelling in Type; category matching and unknown-category behavior belong to the table indexing consumer, not construction.",
                "Exactly one text argument is accepted. Invalid type or shape uses RunMat:table:InvalidArgument. The selector does not inspect any table until used as a subscript.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Data handling and outputs",
            paragraphs: &[
                "Selector metadata is host-only. Numeric and provider-resident values are rejected before provider access.",
                "The result is a scalar host descriptor. Accepted resident payloads are gathered without changing their numeric class, shape, complexity, or exact integer values. Numeric controls follow the restrictions listed above.",
                "Each call returns one descriptor. Calls with no retained output still validate their inputs; requesting more than one output is an error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "standalone-descriptor-contract",
            title: "Select numeric variables",
            program: "T = table([1; 2], {'a'; 'b'}, 'VariableNames', {'Count', 'Label'});\nN = T(:, vartype('numeric'))",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(width(N) == 1);\nassert(isequal(N.Count, [1; 2]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "rowfilter",
            target: BuiltinDocumentationLinkTarget::Builtin("rowfilter"),
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
