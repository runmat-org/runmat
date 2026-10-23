use crate::*;

pub(in crate::catalog::entries::datetime::components) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("month"),
    slug: Some("month"),
    summary: "Extract month components from datetime values in MATLAB and RunMat.",
    description: "`month(t)` extracts calendar-month values from datetime input. Datetime input supports `monthofyear`, `name`, and `shortname`; documented legacy serial-number and date-text forms support an optional input format.",
    keywords: &[
        "month",
        "datetime",
        "date component",
        "calendar month",
    ],
    related: &[
        "datetime",
        "year",
        "day",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and options",
            paragraphs: &[
                "`month(t)` accepts datetime scalars and datetime arrays.",
                "The output shape matches the input shape.",
                "Month values use the standard 1-through-12 numbering.",
                "`month(t,'name')` and `month(t,'shortname')` return shaped cells of character month names for datetime input.",
                "Documented legacy binary64 serial date numbers and date text return double month numbers with matching shape.",
                "Single-precision and all eight typed-integer legacy serial inputs are a gated RunMat extension; exact native values are range-checked before one binary64 serial-date conversion.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host execution and residency",
            paragraphs: &[
                "Documented datetime and legacy serial/text inputs execute on the host. RunMat-only resident inputs gather through their owning provider after compatibility admission.",
                "Documented `month` forms are host-side. RunMat can admit resident legacy numeric input as an extension, gathers it non-destructively through the exact owner, and returns the documented host result.",
                "`month` does not dispatch a GPU kernel. Explicit resident input is a separately gated RunMat extension, while internal automatic residency can gather transparently.",
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
            id: "month-1",
            title: "Extract the month from a scalar datetime",
            program: "t = datetime(2024, 4, 9, 13, 30, 0);\nm = month(t)",
            display_output: Some("m = 4"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(m == 4);",
            },
        },
        BuiltinExample {
            id: "month-2",
            title: "Extract months from a datetime array",
            program: "t = datetime([2024 2025], [1 6], [15 20]);\nm = month(t)",
            display_output: Some("m =\n     1     6"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(m, [1 6]));",
            },
        },
        BuiltinExample {
            id: "month-options",
            title: "Select shaped month names",
            program: "t = datetime([2024 2024], [1 6], [15 20]);\nnames = month(t, 'name');\nshortNames = month(t, 'shortname');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(names, {'January' 'June'}));\nassert(isequal(shortNames, {'Jan' 'Jun'}));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What numbering does `month` use?",
            answer: "It returns January as `1` through December as `12`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `month` preserve the input shape?",
            answer: "Yes. The numeric output has the same size as the input datetime array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `month` run on the GPU?",
            answer: "No. datetime values are host-side objects, so `month` executes on the CPU.",
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
            label: "day",
            target: BuiltinDocumentationLinkTarget::Builtin("day"),
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
                kind: BuiltinEvidenceKind::UnitTest,
                label: "month unit evidence",
                location: "builtins::datetime::tests::month_datetime_name_modes_return_shaped_character_cells",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "month integration evidence",
                location: "crates/runmat-vm/tests/integer_datetime_metadata_and_nonzero_semantics.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
