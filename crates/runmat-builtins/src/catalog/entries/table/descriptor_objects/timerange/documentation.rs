use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timerange"),
    slug: Some("timerange"),
    summary: "Create a timetable row-time range selector object.",
    description: "`timerange(startTime, endTime)` returns a row-time selector with `Start`, `End`, and `Inclusivity` properties.",
    keywords: &[
        "timerange",
        "timetable",
        "row times",
        "selector",
    ],
    related: &[
        "timetable",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The optional third argument is `\"openright\"`, `\"openleft\"`, `\"open\"`, or `\"closed\"`; the default is `\"openright\"`, which includes the start and excludes the end.",
                "Datetime and duration bounds are the compatibility forms. Numeric bounds are available in `runmat` compatibility mode for timetables that use numeric row times.",
                "Numeric integer bounds and numeric integer row times are ordered exactly without conversion to double.",
                "Datetime-unit and event-filter overloads are not yet implemented.",
                "The constructor stores Start and End without validating their shape, class, ordering, or row-time compatibility; those checks belong to indexing the target timetable. Inclusivity is case-insensitive and stored lower-case. Whitespace is trimmed and interval text is stored in canonical lower-case, matching the indexing consumer.",
                "Exactly two bounds and an optional interval type are accepted. One scalar selector is returned; discarded output still performs admission and construction. Invalid arity, nontext inclusivity, and unsupported inclusivity use RunMat:table:InvalidArgument.",
                "Explicit GPU arguments and numeric bounds have separate RunMat-only gates, checked before gathering. Automatically resident inputs follow the existing gather boundary; there is no selector device kernel.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Data handling and outputs",
            paragraphs: &[
                "Selector metadata is host-only.",
                "The result is a scalar host descriptor. Accepted resident payloads are gathered without changing their numeric class, shape, complexity, or exact integer values. Numeric controls follow the restrictions listed above.",
                "Each call returns one descriptor. Calls with no retained output still validate their inputs; requesting more than one output is an error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "standalone-descriptor-contract",
            title: "Select A Half-Open Time Range",
            program: "times = datetime(2026, 7, [1; 15; 31]);\nTT = timetable(times, [10; 20; 30], 'VariableNames', {'Count'});\nfirstTime = datetime(2026, 7, 1);\nlastTime = datetime(2026, 8, 1);\nrows = timerange(firstTime, lastTime);\nJuly = TT(rows, :)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(rows, 'timerange'));\nassert(strcmp(rows.Inclusivity, 'openright'));\nassert(height(July) == 3);\nassert(isequal(July.Count, [10; 20; 30]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "timetable",
            target: BuiltinDocumentationLinkTarget::Builtin("timetable"),
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
