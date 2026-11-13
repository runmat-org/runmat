use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`dateshift(t, 'start', unit)` and `dateshift(t, 'end', unit)` default to the current unit. Supported singular units are `year`, `quarter`, `month`, `week`, `day`, `hour`, `minute`, and `second`.",
            "The end of a year, quarter, month, or week is midnight at the beginning of its last day. The end of a day, hour, minute, or second is exactly the start of the next unit.",
            "`dateshift(t, 'dayofweek', dow)` accepts weekday names or numeric indices from 1 for Sunday through 7 for Saturday. `weekday` and `weekend` select the corresponding day groups.",
            "Rules can be `current`, `next`, `previous`, `nearest`, or integer scalar/arrays. Datetime, weekday, and rule arrays must have matching sizes or be scalar; scalar expansion preserves the non-scalar input shape.",
            "Plural unit aliases, `nearest` as a boundary token, and a custom week-start argument are rejected because they are not part of this public contract.",
            "There is no resident datetime output. Resident arguments are accepted only through the declared RunMat extension and are gathered to a host result.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "dateshift operates on host datetime objects. Resident arguments are compatibility-gated before provider access.",
            "No. Inputs and outputs are host datetime objects. A compatibility-gated resident input is gathered before calendar evaluation.",
            "No GPU kernel is dispatched. MATLAB-compatible mode rejects resident arguments before provider access; RunMat extension mode gathers them and returns a host object.",
            "This pure host operation reads stored datetime serials, not Clock. It may throw or suspend at admitted provider gathers and cooperative checkpoints. Resident controls follow the declared datetime input extension; terminal provider and cancellation errors retain their control-flow authority. There is no device result, GPU kernel, fusion route, or distributed support. Portable Runtime declarations are not backend execution evidence.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Syntax and argument roles",
        paragraphs: &[
            "out = dateshift(t, boundary, unit, rule)",
            "out = dateshift(t, 'dayofweek', weekday, rule)",
            "t must be a datetime object, not relative text or a numeric serial. boundary and calendar unit are scalar text. Unit names are singular and case-insensitive. Day targets additionally accept sun/sunday, mon/monday, tue/tues/tuesday, wed/wednesday, thu/thur/thurs/thursday, fri/friday, and sat/saturday.",
            "The optional rule defaults to current for start/end and next for dayofweek. Numeric weekday and rule controls accept real single/double and all eight native integer classes, with exact integer validation. Weekdays are 1..7; rules must fit int64. Day-target occurrences are additionally bounded to magnitude 200000000 before checked whole-week arithmetic.",
            "There is exactly one output. The input Format is preserved. Scalar expansion preserves the full non-scalar shape, including empty arrays; multiple non-scalar shapes must match exactly.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Calendar arithmetic retains the existing signed serial-date conversion, supported Chrono calendar range, leap-day handling, and binary64 serial-day precision. Nonfinite stored serial elements pass through unchanged. Absolute typed datetime operations require no ambient calendar or Clock fallback.",
            "Empty datetime or numeric controls now preserve their full scalar-expanded empty geometry instead of being mistaken for one output element. Calendar output allocation and copied control vectors are not yet generally managed/cooperatively copied; independent MATLAB parity and complete backend qualification remain incomplete.",
        ],
    },
    BuiltinDocumentationSection { heading: "Finite execution frontier", paragraphs: &["Finite Managed calendar work is currently unsupported: the canonical host-allocation policy refuses before gathers, calendar scratch, or output publication. An initially unbounded call is checked again after suspension. Full managed scratch/output ownership remains an explicit implementation obligation; this safety fence is not Managed completion."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Shift to the Sunday at the start of the current week",
        program: "out = dateshift(datetime(2024, 3, 14), 'start', 'week');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = datetime(2024, 3, 10); assert(isequal(out, expected));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Shift to the next Monday occurrence",
        program: "out = dateshift(datetime(2024, 3, 14), 'dayofweek', 'Monday');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = datetime(2024, 3, 18); assert(isequal(out, expected));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Shift to the end of the current month",
        program: "out = dateshift(datetime(2024, 2, 10), 'end', 'month');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = datetime(2024, 2, 29); assert(isequal(out, expected));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dateshift"),
    slug: Some("dateshift"),
    summary: "Shift datetime values to calendar boundaries or weekday occurrences.",
    description: "`dateshift` implements the public start, end, and day-of-week forms with singular calendar units and optional `current`, `next`, `previous`, `nearest`, or integer occurrence rules. Week calculations begin on Sunday. Weekday indices and integer rules are read from authoritative integer storage before calendar evaluation.",
    keywords: &["dateshift", "datetime", "calendar boundary", "dayofweek", "weekday", "weekend"],
    related: &["datetime", "day"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "When does a week begin?",
            answer: "Sunday, matching the compatibility contract used by MATLAB calendar weeks.",
        },
        BuiltinDocumentationFaq {
            question: "Does end of month mean the last millisecond?",
            answer: "No. For month, quarter, year, and week, it is midnight at the beginning of the final calendar day.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Builtin("datetime") },
        BuiltinDocumentationLink { label: "day", target: BuiltinDocumentationLinkTarget::Builtin("day") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {label: "`crates/runmat-runtime/src/builtins/datetime/mod.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/mod.rs")},
            BuiltinDocumentationLink {label: "`crates/runmat-runtime/src/builtins/datetime/dateshift_builtin.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/dateshift_builtin.rs")},
        ],
        verification: &[
            BuiltinEvidenceReference {kind: BuiltinEvidenceKind::UnitTest, label: "Runtime stored-calendar geometry and precision regression source (not execution evidence)", location: "crates/runmat-runtime/src/builtins/datetime/tests/second_dateshift.rs"},
            BuiltinEvidenceReference {kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity inference source (not execution evidence)", location: "crates/runmat-builtins/src/catalog/entries/datetime/dateshift/inference.rs"},
            BuiltinEvidenceReference {kind: BuiltinEvidenceKind::UnitTest, label: "Existing calendar contract regression source (not execution evidence)", location: "crates/runmat-runtime/src/builtins/datetime/tests.rs"},
        ],
        notes: &["All original substantive prose, syntax, option roles, FAQs, links, and example expressions are conserved. Original displays were unverified claims and are retired in favor of standalone semantic assertions. Historical test pointers are corrected to current source owners; neither references nor authored assertions claim execution. Source-only packet awaits centralized compiler, catalog, documentation and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
