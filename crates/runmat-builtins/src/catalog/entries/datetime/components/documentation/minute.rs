use crate::*;

pub(in crate::catalog::entries::datetime::components) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("minute"),
    slug: Some("minute"),
    summary: "Extract minute components from datetime values in MATLAB and RunMat.",
    description: "`minute(t)` extracts minute-of-hour values from datetime input. The documented legacy forms also accept serial date numbers or date text, with an optional input format for legacy non-datetime input.",
    keywords: &[
        "minute",
        "datetime",
        "time component",
        "clock minute",
    ],
    related: &[
        "datetime",
        "hour",
        "second",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and options",
            paragraphs: &[
                "`minute(t)` accepts datetime scalars and datetime arrays.",
                "The output shape matches the input shape.",
                "Minute values follow the clock minute stored in the datetime.",
                "Documented legacy binary64 serial date numbers and date text return double minute values with matching shape.",
                "Single-precision and all eight typed-integer legacy serial inputs are a gated RunMat extension; exact native values are range-checked before one binary64 serial-date conversion.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host execution and residency",
            paragraphs: &[
                "Documented datetime and legacy serial/text inputs execute on the host. RunMat-only resident inputs gather through their owning provider after compatibility admission.",
                "Documented `minute` forms are host-side. RunMat can admit resident legacy numeric input as an extension, gathers it non-destructively through the exact owner, and returns the documented host double result.",
                "`minute` does not dispatch a GPU kernel. Explicit resident input is a separately gated RunMat extension, while internal automatic residency can gather transparently.",
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
            id: "minute-1",
            title: "Extract the minute from a scalar datetime",
            program: "t = datetime(2024, 4, 9, 13, 30, 5);\nmn = minute(t)",
            display_output: Some("mn = 30"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(mn == 30);",
            },
        },
        BuiltinExample {
            id: "minute-2",
            title: "Extract minutes from a datetime array",
            program: "t = datetime([2024 2024], [4 4], [9 9], [8 17], [15 45], [0 0]);\nmn = minute(t)",
            display_output: Some("mn =\n    15    45"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(mn, [15 45]));",
            },
        },
        BuiltinExample {
            id: "minute-options",
            title: "Extract a minute from legacy text",
            program: "mn = minute(\"2024-04-09 13:30:05\");",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(mn == 30);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What range does `minute` return?",
            answer: "It returns the minute component in the usual `0` through `59` range.",
        },
        BuiltinDocumentationFaq {
            question: "Does `minute` preserve the input shape?",
            answer: "Yes. The numeric output has the same size as the input datetime array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `minute` run on the GPU?",
            answer: "No. datetime values are host-side objects, so `minute` executes on the CPU.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "datetime",
            target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
        },
        BuiltinDocumentationLink {
            label: "hour",
            target: BuiltinDocumentationLinkTarget::Builtin("hour"),
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
                label: "minute unit evidence",
                location: "builtins::datetime::tests::hour_minute_and_month_typed_legacy_serials_cover_every_integer_class",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "minute integration evidence",
                location: "crates/runmat-vm/tests/integer_datetime_metadata_and_nonzero_semantics.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
