use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`second(t)` accepts datetime scalars and datetime arrays.",
            "The output shape matches the input shape.",
            "Returned values are doubles so fractional seconds can be represented.",
            "Passing a non-datetime value raises a datetime-specific error.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "datetime values are host-side objects; `second` extracts the second component on the CPU.",
            "No. `second` reads datetime objects on the CPU and returns ordinary host-side numeric arrays.",
            "`second` does not dispatch GPU kernels; it extracts the second component from CPU-side datetime objects.",
            "This host operation reads only the supplied datetime object; it does not read Clock, gather inputs, or call an acceleration provider. It may throw or suspend at bounded invocation-owned CPU checkpoints. There is no GPU kernel, fusion route, or distributed support. VM/native/Wasm use the canonical portable Runtime binding; this is a source declaration, not backend execution evidence.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Syntax and argument roles",
        paragraphs: &[
            "s = second(t)",
            "t must be a datetime object; relative text, numeric serial values, duration values, and resident numeric values are not converted. Exactly one output is available.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Output precision follows binary64 serial-day storage. Nonfinite serials are rejected by the existing calendar conversion. Native stored serial elements are converted individually without a materialized double copy. Independent MATLAB parity and complete backend qualification remain incomplete.",
        ],
    },
    BuiltinDocumentationSection { heading: "Finite execution frontier", paragraphs: &["Stored datetime second extraction supports finite Managed execution through the canonical exact output owner: the final double payload and full shape are admitted together, input storage remains borrowed and rooted across suspension, and scalar results remain inline. The original owner is checked after suspension and at publication. This closes the second mapper allocation path, not the broader calendar/legacy-text Managed frontier or unexecuted backend qualification."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Extract whole seconds from a scalar datetime",
        program: "t = datetime(2024, 4, 9, 13, 30, 5);\ns = second(t)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(s,'double')); assert(isscalar(s)); assert(abs(s - 5) < 1e-4);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserve fractional seconds",
        program: "t = datetime(2024, 4, 9, 13, 30, 5.25);\ns = second(t)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(s,'double')); assert(isscalar(s)); assert(abs(s - 5.25) < 1e-4);",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("second"),
    slug: Some("second"),
    summary: "Extract second components from datetime values with MATLAB-compatible output shapes.",
    description: "`second` extracts the seconds component from datetime scalars or arrays and returns doubles with the same shape as the input. Fractional seconds are retained to the precision of the stored binary64 serial-date value, not arbitrary nanosecond precision.",
    keywords: &["second", "datetime", "time component", "fractional seconds"],
    related: &["datetime", "hour", "minute"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Can `second` return fractional values?",
            answer: "Yes. RunMat returns fractional seconds as doubles, subject to the precision of the stored binary64 serial-date value.",
        },
        BuiltinDocumentationFaq {
            question: "Does `second` preserve the input shape?",
            answer: "Yes. The numeric output has the same size as the input datetime array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `second` run on the GPU?",
            answer: "No. datetime values are host-side objects, so `second` executes on the CPU.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Builtin("datetime") },
        BuiltinDocumentationLink { label: "hour", target: BuiltinDocumentationLinkTarget::Builtin("hour") },
        BuiltinDocumentationLink { label: "minute", target: BuiltinDocumentationLinkTarget::Builtin("minute") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {label: "`crates/runmat-runtime/src/builtins/datetime/mod.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/mod.rs")},
            BuiltinDocumentationLink {label: "`crates/runmat-runtime/src/builtins/datetime/second_builtin.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/second_builtin.rs")},
        ],
        verification: &[
            BuiltinEvidenceReference {kind: BuiltinEvidenceKind::UnitTest, label: "Runtime stored-calendar geometry and precision regression source (not execution evidence)", location: "crates/runmat-runtime/src/builtins/datetime/tests/second_dateshift.rs"},
            BuiltinEvidenceReference {kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity inference source (not execution evidence)", location: "crates/runmat-builtins/src/catalog/entries/datetime/components/second/inference.rs"},
            BuiltinEvidenceReference {kind: BuiltinEvidenceKind::UnitTest, label: "Existing calendar contract regression source (not execution evidence)", location: "crates/runmat-runtime/src/builtins/datetime/tests.rs"},
        ],
        notes: &["All original substantive prose, syntax, option roles, FAQs, links, and example expressions are conserved. Original displays were unverified claims and are retired in favor of standalone semantic assertions. Historical test pointers are corrected to current source owners; neither references nor authored assertions claim execution. Source-only packet awaits centralized compiler, catalog, documentation and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
