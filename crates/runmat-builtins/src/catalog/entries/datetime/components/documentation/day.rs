use crate::*;

pub(in crate::catalog::entries::datetime::components) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("day"),
    slug: Some("day"),
    summary: "Extract calendar-day numbers or weekday names from datetime and legacy date inputs.",
    description: "`day` supports `dayofmonth`, `dayofweek`, `iso-dayofweek`, `dayofyear`, `name`, and `shortname` for datetime inputs. Numeric modes return host doubles with matching shape; name modes return same-shaped cell arrays of character vectors. Legacy serial-number and text inputs are retained for compatibility and return day-of-month values.",
    keywords: &[
        "day",
        "datetime",
        "date component",
        "day of month",
        "day of week",
        "day of year",
        "ISO weekday",
    ],
    related: &[
        "datetime",
        "year",
        "month",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and options",
            paragraphs: &[
                "`day(t)` defaults to day-of-month. `day(t, dayType)` accepts `dayofmonth`, `dayofweek` with Sunday equal to 1, `iso-dayofweek` with Monday equal to 1, `dayofyear`, `name`, and `shortname`.",
                "Numeric output is double and preserves input shape. Name output is a same-shaped cell array of character vectors; NaT yields NaN numerically and an empty character vector in name modes.",
                "Legacy serial date numbers accept single, double, and all eight integer classes. Legacy date text may provide its datestr-style input format as the second argument; in that dialect lowercase `mm` denotes month, as in `day(\"2021/28/09\", \"yyyy/dd/mm\")`.",
                "Logical and resident numeric legacy inputs are not part of the public compatibility surface. They are rejected in MATLAB-compatible mode before any resident provider access and are available only through declared RunMat extensions.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host execution and residency",
            paragraphs: &[
                "datetime values are host-side objects; `day` extracts the day on the CPU.",
                "No. `day` reads datetime objects on the CPU and returns ordinary host-side numeric arrays.",
                "`day` does not dispatch GPU kernels; it extracts the day component from CPU-side datetime objects.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Owned calendar policy",
            paragraphs: &[
                "Relative date text and timezone-aware text use the invocation-owned calendar service. Hosted execution uses the admitted local platform policy; deterministic deployment uses explicit calendar inputs and fails closed if conversion is unavailable. Numeric and datetime component reads do not acquire an ambient clock.",
            ],
        },
        BuiltinDocumentationSection { heading: "Finite execution frontier", paragraphs: &["Finite Managed calendar work is currently unsupported: the canonical host-allocation policy refuses before gathers, calendar scratch, or output publication. An initially unbounded call is checked again after suspension. Full managed scratch/output ownership remains an explicit implementation obligation; this safety fence is not Managed completion."] },
    ],
    examples: &[
        BuiltinExample {
            id: "day-1",
            title: "Extract the day from a scalar datetime",
            program: "t = datetime(2024, 4, 9, 13, 30, 0);\nd = day(t)",
            display_output: Some("d = 9"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(d == 9);",
            },
        },
        BuiltinExample {
            id: "day-2",
            title: "Extract days from a datetime array",
            program: "t = datetime([2024 2025], [1 6], [15 20]);\nd = day(t)",
            display_output: Some("d =\n    15    20"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(d, [15 20]));",
            },
        },
        BuiltinExample {
            id: "day-3",
            title: "Extract ISO weekday numbers",
            program: "t = datetime(2024, 4, 9);\nd = day(t, 'iso-dayofweek')",
            display_output: Some("d = 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(d == 2);",
            },
        },
        BuiltinExample {
            id: "day-4",
            title: "Extract short weekday names",
            program: "n = day(datetime(2024, 4, 9), 'shortname')",
            display_output: Some("n = {'Tue'}"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(n, {'Tue'}));",
            },
        },
        BuiltinExample {
            id: "day-options",
            title: "Parse a legacy day input format",
            program: "d = day(\"2021/28/09\", \"yyyy/dd/mm\");",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(d == 28);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What day does `day` return?",
            answer: "It returns day-of-month by default. The optional day type selects weekday numbering, ISO weekday numbering, day-of-year, or full and abbreviated weekday names.",
        },
        BuiltinDocumentationFaq {
            question: "Does `day` preserve the input shape?",
            answer: "Yes. The numeric output has the same size as the input datetime array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `day` run on the GPU?",
            answer: "No. datetime values are host-side objects, so `day` executes on the CPU.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "datetime",
            target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
        },
        BuiltinDocumentationLink {
            label: "year",
            target: BuiltinDocumentationLinkTarget::Builtin("year"),
        },
        BuiltinDocumentationLink {
            label: "month",
            target: BuiltinDocumentationLinkTarget::Builtin("month"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/datetime/mod.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/mod.rs"),
            },
            BuiltinDocumentationLink {
                label: "Calendar component implementation",
                target: BuiltinDocumentationLinkTarget::Source(
                    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/component_builtins.rs",
                ),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "day integration evidence",
                location: "crates/runmat-vm/tests/datetime.rs",
            },
        ],
        notes: &["Legacy integration evidence location: runmat-vm/tests/datetime.rs"],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
