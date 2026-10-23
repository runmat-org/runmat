use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "RunMat datetime constructor",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/constructor_builtin.rs",
    ),
};
const TEXT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "RunMat datetime text decoding",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/text_parsing.rs",
    ),
};
const TESTS: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Datetime construction, conversions, modes, arrays, and invalid inputs",
        location: "crates/runmat-runtime/src/builtins/datetime/tests.rs",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "Datetime calls through the VM",
        location: "crates/runmat-vm/tests/datetime.rs",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datetime"),
    slug: Some("datetime"),
    summary: "Create host datetime arrays from text, date vectors, or calendar components.",
    description: "`datetime` constructs host-side datetime objects from date/time text, numeric m-by-3 or m-by-6 date vectors, or calendar components. An explicit `ConvertFrom='datenum'` form reads serial date numbers. `datetime()` reads the current local date and time.",
    keywords: &["datetime", "date", "time", "serial date number", "datenum", "Format", "MATLAB compatibility"],
    related: &["year", "month", "day", "hour", "minute", "second", "datenum", "datevec", "datestr", "calendarDuration", "duration", "string", "char", "disp"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and shapes",
            paragraphs: &[
                "`datetime(Y,M,D)` and `datetime(Y,M,D,H,MI,S)` accept scalar or equally sized arrays of real numeric components. Scalars expand to the non-scalar component shape. A seventh millisecond component is supported; when it is supplied, the second component must be integral. The constructor accepts double, single, and all eight fixed-width integer classes. Date-vector matrices have three or six columns and return one datetime per row.",
                "Calendar components carry into adjacent units. NaN components produce NaT, and infinite components retain an infinite serial marker in the current host representation. Integer values are checked in exact storage before conversion to RunMat's floating serial-date representation.",
                "String scalars, string arrays, character arrays, and cell arrays of character vectors accept common ISO-like date/time text, `dd-MMM-yyyy`, and the relative words `now`, `today`, `tomorrow`, and `yesterday`. Character-array rows yield a datetime column; cell arrays retain their shape. Relative words read the local clock.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Options and compatibility",
            paragraphs: &[
                "`Format` changes the object's display text, not its stored date. `InputFormat` selects a parsing format for text input. `ConvertFrom='datenum'` interprets numeric input as serial date numbers. Other conversion epochs, `Locale`, `PivotYear`, `TimeZone`, `Epoch`, and `TicksPerSecond` are not implemented; unsupported forms report an error rather than silently approximating their semantics.",
                "A one-argument numeric input outside the m-by-3 and m-by-6 date-vector forms is treated as a raw serial date only in RunMat mode. Four- and five-component calls, logical numeric inputs, and resident numeric inputs are also RunMat-only extensions. MATLAB mode rejects these extension forms, including resident input before provider access.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Representation and execution",
            paragraphs: &[
                "A datetime is a host object backed by a serial-date tensor and a `Format` property. There is no resident datetime class or GPU calendar kernel. RunMat mode may gather admitted resident numeric input through its provider owner; the result is still a host object. Construction is a fusion boundary and may suspend for a gather or throw for invalid input.",
                "The no-argument and relative-text forms read the local clock and are nondeterministic. The other supported forms have no filesystem, network, callback, workspace, randomness, or user-interface effect. Native and WebAssembly runtimes use the same host representation; shared injectable clock, timezone, and locale services are not implemented.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Related operations",
            paragraphs: &[
                "Assign `t.Format` to change rendering without changing the stored date. Linear indexing preserves datetime values. Component extractors such as `year`, `month`, `day`, and `hour` return numeric values. Datetime comparison and day arithmetic are supported; subtracting two datetimes returns a numeric day interval. Adding or subtracting a calendarDuration applies month-aware arithmetic, including end-of-month clamping.",
                "`datenum`, `datevec`, `datestr`, `weekday`, `eomday`, `etime`, and `isbetween` cover common legacy serial-date workflows. Business-day functions provide market-calendar calculations; their exact limitations are documented with those functions.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "components", title: "Construct a scalar from calendar components",
            program: "t = datetime(2024,4,9,13,30,0); disp(t);",
            display_output: Some("09-Apr-2024 13:30:00"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isdatetime(t)); assert(year(t)==2024); assert(month(t)==4); assert(day(t)==9); assert(hour(t)==13); assert(minute(t)==30);" },
        },
        BuiltinExample {
            id: "text-array", title: "Parse two text dates",
            program: "t = datetime([\"2024-04-09 13:30:00\"; \"2024-04-10 08:15:00\"]); disp(t);",
            display_output: Some("09-Apr-2024 13:30:00\n10-Apr-2024 08:15:00"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(year(t),[2024;2024])); assert(isequal(day(t),[9;10])); assert(isequal(hour(t),[13;8]));" },
        },
        BuiltinExample {
            id: "cellstr-input-format", title: "Parse a cell array of character vectors",
            program: "dates = {'2024-04-09';'2024-04-10'}; t = datetime(dates,'InputFormat','yyyy-MM-dd'); disp(t);",
            display_output: Some("09-Apr-2024\n10-Apr-2024"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(year(t),[2024;2024])); assert(isequal(day(t),[9;10]));" },
        },
        BuiltinExample {
            id: "extract-components", title: "Extract calendar and clock components",
            program: "t = datetime(2024,4,9,13,30,5); y = year(t); m = month(t); d = day(t); h = hour(t); fprintf('y = %.0f\\nm = %.0f\\nd = %.0f\\nh = %.0f\\n',y,m,d,h);",
            display_output: Some("y = 2024\nm = 4\nd = 9\nh = 13"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            // This date's binary64 serial has spacing 2^-33 days; allow two spacings.
            verification: BuiltinExampleVerification::Assertions { source: "assert(y==2024); assert(m==4); assert(d==9); assert(h==13); assert(minute(t)==30); assert(abs(second(t)-5)<=86400*2^-32);" },
        },
        BuiltinExample {
            id: "serial-datenum", title: "Convert explicit serial date numbers",
            program: "serials = [739351; 739352]; t = datetime(serials,'ConvertFrom','datenum'); disp(t);",
            display_output: Some("09-Apr-2024 00:00:00\n10-Apr-2024 00:00:00"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(datenum(t),serials)); assert(isequal(year(t),[2024;2024])); assert(isequal(month(t),[4;4])); assert(isequal(day(t),[9;10]));" },
        },
        BuiltinExample {
            id: "display-format", title: "Change display format without changing the date",
            program: "t = datetime(2024,4,9,13,30,0); before = datenum(t); t.Format = 'yyyy-MM-dd HH:mm:ss'; disp(t);",
            display_output: Some("2024-04-09 13:30:00"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(datenum(t)==before); assert(strcmp(t.Format,'yyyy-MM-dd HH:mm:ss'));" },
        },
        BuiltinExample {
            id: "month-end", title: "Add a calendar month at month end",
            program: "t = datetime(2024,1,31,'Format','yyyy-MM-dd'); shifted = t + calmonths(1); disp(shifted);",
            display_output: Some("2024-02-29"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(year(shifted)==2024); assert(month(shifted)==2); assert(day(shifted)==29);" },
        },
        BuiltinExample {
            id: "day-arithmetic", title: "Shift a date and subtract two datetimes",
            program: "t0 = datetime(2024,4,9); t1 = t0 + 7; delta = t1 - t0; fprintf('delta = %.0f\\n',delta);",
            display_output: Some("delta = 7"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(day(t1)==16); assert(delta==7);" },
        },
        BuiltinExample {
            id: "serial-helpers", title: "Use serial-date conversion helpers",
            program: "n = datenum(2024,4,9); v = datevec(n); s = datestr(n,'yyyy-mm-dd'); disp(s);",
            display_output: Some("2024-04-09"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(v(1)==2024); assert(v(2)==4); assert(v(3)==9); assert(strcmp(s,'2024-04-09'));" },
        },
        BuiltinExample {
            id: "implicit-serial-extension", title: "Use RunMat's implicit serial-date form",
            program: "serial = datenum(2024,4,9); t = datetime(serial);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isdatetime(t)); assert(datenum(t)==serial);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does `Format` change?", answer: "Only display text. The serial date is unchanged, so comparison and arithmetic refer to the same date and time." },
        BuiltinDocumentationFaq { question: "Can I index a datetime array?", answer: "Yes. Linear `()` indexing returns datetime values. Component functions return numeric values." },
        BuiltinDocumentationFaq { question: "Does datetime run on the GPU?", answer: "No. MATLAB mode rejects resident numeric constructor input before provider access. RunMat mode can gather it and returns a host datetime object." },
        BuiltinDocumentationFaq { question: "How is subtraction interpreted?", answer: "Subtracting another datetime returns a numeric day interval; subtracting a numeric day count shifts the datetime." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB datetime reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/datetime.html") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "disp", target: BuiltinDocumentationLinkTarget::Builtin("disp") },
        BuiltinDocumentationLink { label: "calendarDuration", target: BuiltinDocumentationLinkTarget::Builtin("calendarDuration") },
        BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Builtin("duration") },
        SOURCE,
        TEXT_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, TEXT_SOURCE], verification: TESTS,
        notes: &["Locale, timezone, pivot-year, non-datenum epochs, and general nanosecond fidelity remain unsupported. A shared injectable clock and timezone service is not implemented; clock-dependent forms remain nondeterministic. Native/browser example execution and provider-owner behavior still need integrated qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
