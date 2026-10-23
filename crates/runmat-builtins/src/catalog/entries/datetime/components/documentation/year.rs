use crate::*;

pub(in crate::catalog::entries::datetime::components) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("year"),
    slug: Some("year"),
    summary: "Extract calendar year components from datetime values with MATLAB-compatible output shaping.",
    description: "`year(t)` extracts year numbers from datetime values and retained legacy date inputs, returning double values with the input shape.",
    keywords: &[
        "year",
        "datetime",
        "date component",
        "calendar year",
    ],
    related: &[
        "datetime",
        "month",
        "day",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and options",
            paragraphs: &[
                "`year(t)` accepts datetime scalars and arrays. The optional year type is `'iso'` or `'gregorian'`.",
                "For compatibility with existing code, serial date numbers and text dates are also accepted; an optional date format can be supplied for legacy input.",
                "The output shape matches the input shape.",
                "The returned values are numeric doubles, even when the years are mathematically integers.",
                "RunMat mode additionally accepts single-precision and native integer legacy serial-date storage.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host execution and residency",
            paragraphs: &[
                "datetime values are host-side objects; `year` reads the stored serial-date data on the CPU.",
                "No. `year` reads datetime objects on the CPU and returns ordinary host-side numeric arrays.",
                "`year` does not dispatch GPU kernels. If the datetime was produced from gathered numeric data, the result still remains a CPU numeric array.",
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
            id: "year-1",
            title: "Extract the year from a scalar datetime",
            program: "t = datetime(2024, 4, 9, 13, 30, 0);\ny = year(t)",
            display_output: Some("y = 2024"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(y == 2024);",
            },
        },
        BuiltinExample {
            id: "year-2",
            title: "Extract years from a datetime array",
            program: "t = datetime([2024 2025], [1 6], [15 20]);\ny = year(t)",
            display_output: Some("y =\n     2024     2025"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(y, [2024 2025]));",
            },
        },
        BuiltinExample {
            id: "year-options",
            title: "Select ISO and Gregorian year modes",
            program: "t = datetime(2024, 4, 9);\nyIso = year(t, 'iso');\nyGregorian = year(t, 'gregorian');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(yIso == 2024 && yGregorian == 2024);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `year` preserve the input shape?",
            answer: "Yes. A scalar datetime returns a scalar double, and datetime arrays return numeric arrays of the same size.",
        },
        BuiltinDocumentationFaq {
            question: "Does `year` return integers?",
            answer: "It returns MATLAB-style doubles whose values are whole-number years.",
        },
        BuiltinDocumentationFaq {
            question: "Does `year` run on the GPU?",
            answer: "No. datetime values are host-side objects, so `year` executes on the CPU.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "datetime",
            target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
        },
        BuiltinDocumentationLink {
            label: "month",
            target: BuiltinDocumentationLinkTarget::Builtin("month"),
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
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "year integration evidence",
                location: "crates/runmat-vm/tests/datetime.rs",
            },
        ],
        notes: &["Legacy integration evidence location: runmat-ignition/tests/datetime.rs"],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
