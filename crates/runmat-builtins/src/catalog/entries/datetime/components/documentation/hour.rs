use crate::*;

pub(in crate::catalog::entries::datetime::components) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hour"),
    slug: Some("hour"),
    summary: "Extract hour components from datetime values in MATLAB and RunMat.",
    description: "`hour(t)` extracts hour-of-day values from datetime input. The documented legacy forms also accept serial date numbers or date text, with an optional input format for legacy non-datetime input.",
    keywords: &[
        "hour",
        "datetime",
        "time component",
        "clock hour",
    ],
    related: &[
        "datetime",
        "minute",
        "second",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and options",
            paragraphs: &[
                "`hour(t)` accepts datetime scalars and datetime arrays.",
                "The output shape matches the input shape.",
                "Hour values are returned in 24-hour clock form.",
                "Documented legacy binary64 serial date numbers and date text return double hour values with matching shape.",
                "Single-precision and all eight typed-integer legacy serial inputs are a gated RunMat extension; exact native values are range-checked before one binary64 serial-date conversion.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host execution and residency",
            paragraphs: &[
                "Documented datetime and legacy serial/text inputs execute on the host. RunMat-only resident inputs gather through their owning provider after compatibility admission.",
                "Documented `hour` forms are host-side. RunMat can admit resident legacy numeric input as an extension, gathers it non-destructively through the exact owner, and returns the documented host double result.",
                "`hour` does not dispatch a GPU kernel. Explicit resident input is a separately gated RunMat extension, while internal automatic residency can gather transparently.",
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
            id: "hour-1",
            title: "Extract the hour from a scalar datetime",
            program: "t = datetime(2024, 4, 9, 13, 30, 5);\nh = hour(t)",
            display_output: Some("h = 13"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(h == 13);",
            },
        },
        BuiltinExample {
            id: "hour-2",
            title: "Extract hours from a datetime array",
            program: "t = datetime([2024 2024], [4 4], [9 9], [8 17], [15 45], [0 0]);\nh = hour(t)",
            display_output: Some("h =\n     8    17"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(h, [8 17]));",
            },
        },
        BuiltinExample {
            id: "hour-options",
            title: "Extract an hour from legacy text",
            program: "h = hour(\"2024-04-09 13:30:05\");",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(h == 13);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What clock convention does `hour` use?",
            answer: "It returns 24-hour clock values such as `0`, `13`, or `23`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `hour` preserve the input shape?",
            answer: "Yes. The numeric output has the same size as the input datetime array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `hour` run on the GPU?",
            answer: "No. datetime values are host-side objects, so `hour` executes on the CPU.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "datetime",
            target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
        },
        BuiltinDocumentationLink {
            label: "minute",
            target: BuiltinDocumentationLinkTarget::Builtin("minute"),
        },
        BuiltinDocumentationLink {
            label: "second",
            target: BuiltinDocumentationLinkTarget::Builtin("second"),
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
                label: "hour unit evidence",
                location: "builtins::datetime::tests::hour_minute_and_month_typed_legacy_serials_cover_every_integer_class",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "hour integration evidence",
                location: "crates/runmat-vm/tests/integer_datetime_metadata_and_nonzero_semantics.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
