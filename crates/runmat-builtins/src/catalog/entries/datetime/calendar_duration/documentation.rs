use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Calendar-duration implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/calendar_duration_builtins.rs",
    ),
};
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Calendar-duration construction, extraction, and validation",
    location: "crates/runmat-runtime/src/builtins/datetime/tests.rs",
}];
const EXECUTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Execution and placement",
    paragraphs: &[
        "Calendar units use a host calendarDuration object with separate month and day tensors. Construction and extraction run on the host; they do not launch a GPU calendar kernel and form a fusion boundary. RunMat mode can gather resident numeric input through its provider owner. MATLAB mode rejects that extension before provider access.",
        "These functions work in native and WebAssembly runtimes. They do not use files, the network, the clock, callbacks, or workspace state. Resident gathers may suspend and fail; iscalendarduration only inspects the value class and does not gather it.",
    ],
};

macro_rules! unit_doc {
    ($constant:ident, $name:literal, $unit:literal, $program:literal, $oracle:literal, $url:literal) => {
        pub(super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($name),
            slug: Some($name),
            summary: concat!("Create calendar durations in ", $unit, " or extract whole ", $unit, "."),
            description: concat!("`", $name, "(X)` creates a calendarDuration array when X is a real numeric or logical array of whole values. For calendarDuration input, it returns double whole-unit counts with the same shape. Month and day components remain separate; converting a duration to one unit does not approximate months as fixed-length days."),
            keywords: &[$name, "calendarDuration", "datetime", "calendar units"],
            related: &["calendarDuration", "iscalendarduration"],
            sections: &[
                BuiltinDocumentationSection { heading: "Input and result", paragraphs: &[
                    "Numeric inputs must contain finite integers; fractional units are rejected. Logical values represent zero or one. The output has the input's array shape. Extracting whole units truncates toward zero and returns double; month-based units use the stored month component, while day/week units use the stored day component.",
                    "Calendar duration objects preserve calendar-relative months separately from days. This matters at month ends and across leap years; use datetime arithmetic to resolve a duration against a particular date.",
                ] },
                EXECUTION,
            ],
            examples: &[BuiltinExample {
                id: "roundtrip", title: concat!("Construct and extract ", $unit),
                program: $program, display_output: None,
                compatibility: BuiltinExampleCompatibility::Matlab,
                harness: BuiltinExampleHarness::Portable,
                fixture: BuiltinExampleFixture::None,
                requirements: BuiltinExampleRequirements::NONE,
                verification: BuiltinExampleVerification::Assertions { source: $oracle },
            }],
            example_exemption: None,
            faqs: &[BuiltinDocumentationFaq {
                question: "Can I construct a fractional calendar unit?",
                answer: "No. Numeric constructor inputs must be finite whole numbers. A calendarDuration can still include fractional seconds when created from components.",
            }],
            links: &[
                BuiltinDocumentationLink { label: "MATLAB reference", target: BuiltinDocumentationLinkTarget::External($url) },
                BuiltinDocumentationLink { label: "calendarDuration", target: BuiltinDocumentationLinkTarget::Builtin("calendarDuration") },
                SOURCE,
            ],
            media: &[],
            evidence: BuiltinDocumentationEvidence {
                implementation: &[SOURCE], verification: TESTS,
                notes: &["The host calendar-duration representation does not implement all timezone, display-format, tall-array, or distributed-array forms documented by MATLAB; native and browser examples need integrated replay."],
            },
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Partial),
        };
    };
}

unit_doc!(
    CALDAYS,
    "caldays",
    "days",
    "c = caldays([2 3]); n = caldays(c);",
    "assert(iscalendarduration(c)); assert(isequal(n,[2 3]));",
    "https://www.mathworks.com/help/matlab/ref/calendarduration.caldays.html"
);
unit_doc!(
    CALWEEKS,
    "calweeks",
    "weeks",
    "c = calweeks([2 3]); n = calweeks(c);",
    "assert(iscalendarduration(c)); assert(isequal(n,[2 3]));",
    "https://www.mathworks.com/help/matlab/ref/calendarduration.calweeks.html"
);
unit_doc!(
    CALMONTHS,
    "calmonths",
    "months",
    "c = calmonths([2 3]); n = calmonths(c);",
    "assert(iscalendarduration(c)); assert(isequal(n,[2 3]));",
    "https://www.mathworks.com/help/matlab/ref/calendarduration.calmonths.html"
);
unit_doc!(
    CALQUARTERS,
    "calquarters",
    "quarters",
    "c = calquarters([2 3]); n = calquarters(c);",
    "assert(iscalendarduration(c)); assert(isequal(n,[2 3]));",
    "https://www.mathworks.com/help/matlab/ref/calendarduration.calquarters.html"
);
unit_doc!(
    CALYEARS,
    "calyears",
    "years",
    "c = calyears([2 3]); n = calyears(c);",
    "assert(iscalendarduration(c)); assert(isequal(n,[2 3]));",
    "https://www.mathworks.com/help/matlab/ref/calendarduration.calyears.html"
);

pub(super) const CALENDARDURATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("calendarDuration"), slug: Some("calendarDuration"),
    summary: "Create calendar-relative durations from calendar components.",
    description: "`calendarDuration(Y,M,D)` or `calendarDuration(Y,M,D,H,MI,S)` constructs calendar-duration values with separately stored month and day components. A numeric matrix with three or six columns is also accepted; each row makes one duration. The current implementation preserves component shape and scalar expansion, but display Format and duration-valued time components are not yet implemented.",
    keywords: &["calendarDuration", "caldays", "calweeks", "calmonths", "calquarters", "calyears", "datetime", "MATLAB compatibility", "calendar arithmetic"],
    related: &["caldays", "calmonths", "datetime", "duration"],
    sections: &[
        BuiltinDocumentationSection { heading: "Components and arrays", paragraphs: &[
            "Years, months, days, hours, and minutes must be finite integers. Seconds may be fractional. Non-scalar component arrays must have matching sizes; scalar components expand to that size. The month field stores twelve times years plus months, while day/time fields remain separate from months.",
            "A single three- or six-column numeric matrix supplies one calendar duration per row. RunMat mode also accepts `calendarDuration()` as zero, a single numeric array as day counts, a calendarDuration copy input, and four- or five-component calls; those shorthand forms are not part of the documented MATLAB constructor contract and are rejected in MATLAB mode. Logical constructor components are also a RunMat-only extension.",
            "RunMat supports calendarDuration addition and subtraction with another calendarDuration, and month-aware addition or subtraction with datetime. Datetime arithmetic applies months before days and clamps dates that do not exist in the destination month. The current object model does not yet reproduce every timezone or display-format behavior of MATLAB calendarDuration.",
        ] }, EXECUTION,
    ],
    examples: &[
        BuiltinExample { id: "components", title: "Construct from three calendar components", program: "c = calendarDuration(1,2,3); m = calmonths(c); d = caldays(c);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscalendarduration(c)); assert(m == 14); assert(d == 3);" } },
        BuiltinExample { id: "matrix", title: "Construct one duration per matrix row", program: "c = calendarDuration([1 2 3; 0 4 5]); m = calmonths(c); d = caldays(c);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(m,[14;4])); assert(isequal(d,[3;5]));" } },
        BuiltinExample { id: "month-end", title: "Add a calendar month at the end of January", program: "t = datetime(2024,1,31); shifted = t + calmonths(1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(year(shifted)==2024); assert(month(shifted)==2); assert(day(shifted)==29);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why are months and days stored separately?", answer: "The number of days in a month depends on the date to which the duration is applied. Month-based extraction therefore reads the month component and day-based extraction reads the day component." },
        BuiltinDocumentationFaq { question: "How is calendarDuration different from duration?", answer: "`duration` stores fixed elapsed time. `calendarDuration` stores calendar months and days, so adding one month to January 31 lands on the final day of February when that day does not exist." },
        BuiltinDocumentationFaq { question: "Does calendarDuration run on the GPU?", answer: "No calendar kernel is launched. In RunMat mode, resident numeric input can be gathered to a host calendarDuration object; MATLAB mode rejects that extension before provider access." },
    ],
    links: &[BuiltinDocumentationLink { label: "MATLAB calendarDuration reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/calendarduration.html") }, BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Builtin("duration") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Format options, duration-valued time components, tall/distributed arrays, and timezone-sensitive arithmetic remain unsupported. These are explicit partial-coverage limits."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ISCALENDARDURATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("iscalendarduration"), slug: Some("iscalendarduration"),
    summary: "Test whether a value is a calendarDuration array.",
    description: "`iscalendarduration(X)` returns a host logical scalar. It checks the exact class identity of the value, not its internal numeric components; a numeric array or another object returns false.",
    keywords: &["iscalendarduration", "calendarDuration", "predicate"],
    related: &["calendarDuration", "caldays"],
    sections: &[EXECUTION],
    examples: &[BuiltinExample { id: "class-check", title: "Distinguish a calendar duration from a number", program: "c = caldays(2); a = iscalendarduration(c); b = iscalendarduration(2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(a); assert(~b);" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB iscalendarduration reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/iscalendarduration.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["The predicate is synchronous and does not access provider data."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
