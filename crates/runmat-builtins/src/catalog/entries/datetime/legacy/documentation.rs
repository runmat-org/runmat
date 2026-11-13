use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Legacy datetime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/legacy_builtins.rs"),
};
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Date conversion, output arity, calendars, and interval boundaries",
    location: "crates/runmat-runtime/src/builtins/datetime/tests.rs",
}];
const EXECUTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Execution and placement",
    paragraphs: &[
        "The current date implementation evaluates these operations on the host. Resident numeric input is a RunMat extension and is rejected in MATLAB mode before provider access. In RunMat mode it is gathered through its provider owner; no date GPU kernel or fused operation is used. A provider gather can suspend or fail. The returned numeric, logical, or character value is host-resident.",
        "These functions do not read files, access the network, call user callbacks, or mutate workspace state. The relative text values now, today, tomorrow, and yesterday read the local clock; datenum() also reads it. Results from those forms depend on the machine's local time setting.",
    ],
};

pub(super) const DATENUM: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datenum"), slug: Some("datenum"),
    summary: "Convert supported date inputs to serial day numbers.",
    description: "`datenum` returns double serial day numbers. It accepts datetime values, supported date text, three- or six-column date vectors, or three to six numeric components with scalar expansion. A zero-argument call reads the current local time. A numeric tensor that is not recognized as a date vector is interpreted as serial dates in the current RunMat implementation.",
    keywords: &["datenum", "datetime", "serial date", "date vector"],
    related: &["datevec", "datestr", "datetime"],
    sections: &[
        BuiltinDocumentationSection { heading: "Date-vector and component forms", paragraphs: &[
            "Each row of an N-by-3 or N-by-6 matrix produces one serial date. Three-component rows supply year, month, and day; six-component rows add hour, minute, and second. Non-scalar component arrays must have matching shapes; scalar components expand. A single result is a double scalar and multiple results form an array.",
            "RunMat validates calendar dates and time components. Four- and five-component forms, logical components, and single or integer numeric input beyond the documented double forms require RunMat mode. The current date model stores instants as floating serial days, so values that require finer precision than that representation may round. Text parsing covers common ISO and day-month-name forms but not all MATLAB formatIn, PivotYear, locale, or time-zone forms.",
        ] }, EXECUTION,
    ],
    examples: &[
        BuiltinExample { id: "components", title: "Convert components to a serial date", program: "n = datenum(2024,3,14);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "v = datevec(n); assert(isequal(v(1:3), [2024 3 14]));" } },
        BuiltinExample { id: "date-vector-rows", title: "Convert two date-vector rows", program: "n = datenum([2024 3 14; 2024 3 15]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(n), [2 1])); assert(abs(n(2)-n(1)-1)<1e-9);" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does datenum account for time zones?", answer: "The current RunMat datetime object has no full time-zone model. Calls without a date and relative text use the machine's local clock." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB datenum reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/datetime.datenum.html") }, BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Builtin("datetime") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Full formatIn, PivotYear, locale, timezone, carryover, and duration semantics are not implemented; integrated native/browser replay remains required."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const DATEVEC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datevec"), slug: Some("datevec"),
    summary: "Return six calendar components for each date.",
    description: "`v = datevec(D)` returns an N-by-6 double matrix with year, month, day, hour, minute, and second in each row. Requested outputs `[Y,M,D,H,MI,S]` split those columns into matching N-by-1 values; with one requested output, the complete six-column matrix is returned.",
    keywords: &["datevec", "datenum", "datetime", "calendar components"], related: &["datenum", "datetime"],
    sections: &[BuiltinDocumentationSection { heading: "Inputs and result", paragraphs: &[
        "RunMat accepts numeric serial dates, datetime objects, and supported date text. Numeric arrays are flattened into one date vector per element. Single-precision and integer serial input require RunMat mode. Component seconds may contain a fractional value. More than six outputs are not supported.",
        "The host datetime object stores serial values internally; static analysis cannot recover its hidden array size merely from the scalar object wrapper. Inference leaves row count dynamic unless the input's numeric or text shape proves it.",
    ] }, EXECUTION],
    examples: &[
        BuiltinExample { id: "matrix", title: "Return a complete date vector", program: "n = datenum(2024,3,14); v = datevec(n);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(v), [1 6])); assert(isequal(v(1:3), [2024 3 14]));" } },
        BuiltinExample { id: "components", title: "Request the year and month separately", program: "n = datenum(2024,3,14); [Y,M] = datevec(n);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(Y == 2024 && M == 3);" } },
    ], example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Why does one output differ from two outputs?", answer: "One output is the complete N-by-6 matrix. Two or more outputs select individual columns." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB datevec reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/datetime.datevec.html") }, BuiltinDocumentationLink { label: "datenum", target: BuiltinDocumentationLinkTarget::Builtin("datenum") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Legacy text format, PivotYear, duration, timezone, and negative serial special cases remain partial; integrated output-arity replay is required."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const DATESTR: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datestr"), slug: Some("datestr"),
    summary: "Format supported dates as character rows.",
    description: "`datestr(D)` renders supported datetime, serial-date, or date-vector input as a character array with one row per date. An optional text format or numeric format code selects the displayed calendar fields; rows are padded to equal width.",
    keywords: &["datestr", "date format", "date string", "datenum"], related: &["datenum", "datetime"],
    sections: &[BuiltinDocumentationSection { heading: "Supported formatting", paragraphs: &[
        "The datestr legacy format language uses lowercase mm for month and uppercase MM for minute; it differs from datetime display formats. Numeric double format codes -1 through 31 are supported. The default omits the time field only when every finite date in the input is exactly midnight; code 0 always includes it. Repeated or incompatible fields are rejected. Locale selection and PivotYear are not yet implemented. A three- or six-column numeric matrix is interpreted as date vectors; other numeric shapes are serial dates. Single-precision and integer numeric date inputs require RunMat mode.",
        "The result is a host character array, not a string array. Its number of rows equals the number of input dates, while the width depends on the format and values.",
    ] }, EXECUTION],
    examples: &[BuiltinExample { id: "iso-date", title: "Format a serial date", program: "s = datestr(datenum(2024,3,14), 'yyyy-mm-dd');", display_output: Some("2024-03-14"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s, '2024-03-14'));" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Does datestr return a string array?", answer: "No. It returns a character row or a padded character matrix." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB datestr reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/datetime.datestr.html") }, BuiltinDocumentationLink { label: "datenum", target: BuiltinDocumentationLinkTarget::Builtin("datenum") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Legacy datestr tokens and numeric format codes are implemented; integrated examples remain unexecuted in this source packet."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const WEEKDAY: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("weekday"), slug: Some("weekday"),
    summary: "Return Sunday-first weekday numbers and optional English names.",
    description: "`weekday(D)` returns double day numbers from 1 (Sunday) to 7 (Saturday). `[number,name] = weekday(D)` also returns abbreviated English day names as character rows. `weekday(D,'long')` requests full English names; `'short'` is the default.",
    keywords: &["weekday", "day of week", "calendar", "datetime"], related: &["day", "datetime"],
    sections: &[BuiltinDocumentationSection { heading: "Day-name forms", paragraphs: &[
        "The first output follows the input date array's shape. The optional second output has one character row per date, padded to its longest name. English short and long forms are supported; locale/language arguments and locale-dependent names are not yet implemented.",
        "Numeric serial dates, datetime objects, and supported date text are accepted. A resident numeric input is gathered to the host; an invalid DayForm is rejected before provider access.",
    ] }, EXECUTION],
    examples: &[BuiltinExample { id: "names", title: "Number and English weekday name", program: "n = datenum(2024,3,14); [d,name] = weekday(n);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(d == 5); assert(strcmp(name, 'Thurs'));" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Which day is 1?", answer: "Sunday is 1 and Saturday is 7." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB weekday reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/weekday.html") }, BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Builtin("datetime") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Language and local-locale name forms remain partial; integrated two-output replay is pending."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const EOMDAY: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("eomday"), slug: Some("eomday"),
    summary: "Return the last day number for a calendar month.",
    description: "`eomday(Y,M)` returns a host double array of month lengths for year and month inputs. Scalar values expand over an array input; non-scalar inputs must have matching sizes. Calendar validation includes leap years.",
    keywords: &["eomday", "end of month", "leap year", "calendar"], related: &["datetime", "day"],
    sections: &[BuiltinDocumentationSection { heading: "Calendar values", paragraphs: &[
        "Years must be finite whole values within the current date range, and months must be whole values from 1 through 12. Invalid dates or incompatible input shapes raise an error rather than wrapping or silently truncating.",
    ] }, EXECUTION],
    examples: &[BuiltinExample { id: "leap-year", title: "February in a leap year", program: "d = eomday(2024,2);", display_output: Some("29"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(d == 29);" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Does February account for leap years?", answer: "Yes. For example, eomday(2024,2) returns 29." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB eomday reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/eomday.html") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Array and edge-date behavior still need integrated native/browser replay."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ETIME: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("etime"), slug: Some("etime"),
    summary: "Find elapsed seconds between date vectors.",
    description: "`etime(t2,t1)` converts three- or six-component date-vector rows to serial dates and returns `(t2-t1)*86400` seconds. A single date-vector row expands over the rows of the other input; non-scalar row counts must match.",
    keywords: &["etime", "elapsed seconds", "datevec", "datetime"], related: &["datevec", "seconds"],
    sections: &[BuiltinDocumentationSection { heading: "Precision and shapes", paragraphs: &[
        "The output is host double. Seconds are calculated from RunMat's floating serial-day representation; small intervals at large date magnitudes can therefore lose subsecond precision. This function accepts date vectors, not arbitrary datetime objects or timezone-aware instants.",
    ] }, EXECUTION],
    examples: &[BuiltinExample { id: "one-day", title: "Elapsed seconds over one day", program: "dt = etime([2024 1 2 0 0 0], [2024 1 1 0 0 0]);", display_output: Some("86400"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(dt == 86400);" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "What is the unit?", answer: "Seconds, including a fractional part when the represented date vectors permit it." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB etime reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/etime.html") }, BuiltinDocumentationLink { label: "datevec", target: BuiltinDocumentationLinkTarget::Builtin("datevec") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Serial-day precision is a known limit; integrated array and cross-platform checks remain required."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ISBETWEEN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isbetween"), slug: Some("isbetween"),
    summary: "Test whether dates fall inclusively between two bounds.",
    description: "`isbetween(t,lower,upper)` compares supported date values against inclusive lower and upper bounds. The result is a host logical scalar or array with the broadcast shape of its inputs.",
    keywords: &["isbetween", "datetime", "date range", "logical"], related: &["datetime", "datenum"],
    sections: &[BuiltinDocumentationSection { heading: "Current supported form", paragraphs: &[
        "RunMat accepts datetime values in both modes; supported date text and numeric serial dates are RunMat-only convenience forms. Array inputs use implicit expansion where shapes are compatible. Endpoint inclusivity options and additional duration-specific forms are not implemented; the current form always includes both bounds.",
    ] }, EXECUTION],
    examples: &[BuiltinExample { id: "inclusive", title: "Include both date bounds", program: "d = datenum(2024,3,14); inside = isbetween(d,d,d);", display_output: Some("true"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(inside)); assert(inside);" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Are the endpoints included?", answer: "Yes. Equality to either bound returns true in the current supported form." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB isbetween reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/datetime.isbetween.html") }, BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Builtin("datetime") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Endpoint inclusivity options and full datetime/duration class parity remain partial; integrated predicate replay is required."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
