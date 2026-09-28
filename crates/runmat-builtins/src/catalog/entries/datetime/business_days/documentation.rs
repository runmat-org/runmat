use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "RunMat business-day implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/business_day_builtins.rs",
    ),
};
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Business-day calendar, types, modes, shapes, and rejected inputs",
    location: "crates/runmat-runtime/src/builtins/datetime/tests.rs",
}];
const EXECUTION: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection {
    heading: "Execution and placement",
    paragraphs: &[
        "RunMat evaluates calendar rules on the host. The call is a fusion boundary. In RunMat compatibility mode, admitted resident operands are gathered before calculation; MATLAB compatibility mode rejects them before provider access. There is no GPU calendar kernel.",
        "Single-precision and fixed-width integer legacy serial dates, logical values in numeric date or control positions, and resident numeric inputs are RunMat-only extensions. Logical Weekend masks remain accepted in MATLAB compatibility mode.",
        "The calendar functions are available in the native runtime and WebAssembly. No filesystem, network, user callback, or worker service is required. The zero-argument `holidays` form reads the local clock.",
    ],
}];

pub(super) const ISBUSDAY: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isbusday"), slug: Some("isbusday"),
    summary: "Identify business dates in a date array.",
    description: "`isbusday(Date)` returns a logical value for each date. Saturday, Sunday, and the selected holidays are nonbusiness dates by default. An optional seven-element Sunday-first `Weekend` mask changes which weekdays are excluded.",
    keywords: &["isbusday", "business day", "weekend", "holiday", "datetime"],
    related: &["busdays", "holidays"],
    sections: &[
        BuiltinDocumentationSection { heading: "Calendar", paragraphs: &[
            "An omitted or empty holiday argument uses RunMat's built-in market-holiday calculation. A single `NaN` holiday argument excludes no holidays. The built-in calendar does not yet include the full historical NYSE special-closure record.",
            "The `Weekend` mask contains seven 0/1 values starting on Sunday; `[1 0 0 0 0 0 1]` is the default.",
        ] },
        EXECUTION[0],
    ],
    examples: &[BuiltinExample {
        id: "logical-mask", title: "Check three dates",
        program: "dates = [datenum(2024,1,1) datenum(2024,1,2) datenum(2024,1,6)];\nmask = isbusday(dates);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(mask)); assert(isequal(mask,logical([0 1 0])));" },
    }, BuiltinExample {
        id: "custom-weekend", title: "Allow Saturday trading",
        program: "d = datenum(2003,3,1); open = isbusday(d,[],[1 0 0 0 0 0 0]);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(open);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can Saturday be a business day?", answer: "Yes. Supply a Sunday-first `Weekend` mask that marks Saturday as 0, for example `[1 0 0 0 0 0 0]`." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB isbusday reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/isbusday.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["The runtime currently approximates the market calendar rather than carrying the complete historical exchange closure dataset."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const BUSDAYS: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("busdays"), slug: Some("busdays"),
    summary: "Generate business dates over a calendar period.",
    description: "`busdays(StartDate,EndDate)` generates a column of business dates. The optional `bdmode` chooses daily, weekly, monthly, quarterly, semiannual, or annual selection, and a fourth argument supplies holidays.",
    keywords: &["busdays", "business calendar", "trading dates", "bdmode"],
    related: &["isbusday", "fbusdate", "lbusdate"],
    sections: &[
        BuiltinDocumentationSection { heading: "Period and output", paragraphs: &[
            "The default mode is daily. Modes 1 through 6, or their frequency names, select daily through annual periods. For a nondaily period, RunMat chooses the last business date on or before the calendar period end.",
            "A datetime start, end, or holiday argument produces datetime dates; otherwise the result contains serial date numbers. The result is a column vector, including an empty column when no business date falls in a daily range.",
            "RunMat currently caps the calendar span at one million days and uses a computed market-holiday calendar rather than a complete exchange closure history.",
        ] },
        EXECUTION[0],
    ],
    examples: &[BuiltinExample {
        id: "daily-range", title: "List dates across a weekend",
        program: "first = datenum(2024,1,5); last = datenum(2024,1,8);\nd = busdays(first,last);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,[first;last]));" },
    }, BuiltinExample {
        id: "monthly-period", title: "Select monthly business dates",
        program: "d = busdays(datenum(2024,1,1),datenum(2024,2,29),3);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,[datenum(2024,1,31);datenum(2024,2,29)]));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How are custom holidays supplied?", answer: "Pass `bdmode` as the third argument and the holiday dates as the fourth argument." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB busdays reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/busdays.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["The periodicity and market-calendar calculations need product-level example and browser verification before a complete-parity claim."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const DAYS252BUS: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("days252bus"), slug: Some("days252bus"),
    summary: "Count business dates between two dates.",
    description: "`days252bus(StartDate,EndDate)` counts nonweekend, nonholiday dates in the selected interval. Scalar dates broadcast over a compatible date array.",
    keywords: &["days252bus", "business days", "day count", "finance"],
    related: &["daysdif", "isbusday"],
    sections: &[BuiltinDocumentationSection { heading: "Calendar and sign", paragraphs: &[
        "The result is a double scalar or array. Reversed dates yield a negative count. An omitted or empty holiday vector uses the built-in computed market calendar; a single `NaN` value omits holidays.",
        "RunMat's computed calendar does not include every historical special market closure, so dates requiring that record should use an explicit holiday vector.",
    ] }, EXECUTION[0]],
    examples: &[BuiltinExample {
        id: "weekend-count", title: "Count dates across a weekend",
        program: "a = datenum(2024,1,5); b = datenum(2024,1,8); n = days252bus(a,b);",
        display_output: Some("n = 2"), compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 2);" },
    }, BuiltinExample {
        id: "custom-holiday-count", title: "Exclude an explicit holiday",
        program: "a = datenum(2024,1,5); b = datenum(2024,1,8); n = days252bus(a,b,b);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 1);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can I use my own trading calendar?", answer: "Yes. Pass its holiday-date vector as the third argument." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB days252bus reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/days252bus.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Default historical market closures remain a known calendar-data gap."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const DAYSDIF: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("daysdif"), slug: Some("daysdif"),
    summary: "Calculate date differences under a selected day-count basis.",
    description: "`daysdif(StartDate,EndDate)` returns elapsed calendar days. Basis 1 applies the currently implemented 30/360 SIA rule. Other public basis codes are rejected explicitly until their algorithms are implemented.",
    keywords: &["daysdif", "date difference", "30/360", "day-count basis"],
    related: &["days252bus"],
    sections: &[BuiltinDocumentationSection { heading: "Supported bases", paragraphs: &[
        "Basis 0 counts actual elapsed dates and excludes the start date. Basis 1 uses RunMat's current 30/360 SIA calculation. Basis must be a finite scalar integer; values 2 through 13 are not yet implemented and return an error rather than silently reverting to Basis 0.",
        "Start and end date arrays may match in size or one side may be scalar. The result is double and retains the broadcast result shape.",
    ] }, EXECUTION[0]],
    examples: &[BuiltinExample {
        id: "actual-days", title: "Find the elapsed number of calendar days",
        program: "a = datenum(2024,1,5); b = datenum(2024,1,8); n = daysdif(a,b);",
        display_output: Some("n = 3"), compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 3);" },
    }, BuiltinExample {
        id: "sia-february-end", title: "Use the 30/360 SIA basis at February end",
        program: "n = daysdif(datenum(2023,2,28),datenum(2023,3,1),1);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 1);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Which basis values work today?", answer: "RunMat implements 0 and 1. It rejects other basis values explicitly." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB daysdif reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/daysdif.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Public basis codes 2 through 13 remain unimplemented; this entry is intentionally Partial."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const FBUSDATE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fbusdate"), slug: Some("fbusdate"),
    summary: "Find the first business date of a month.",
    description: "`fbusdate(Year,Month)` returns a serial date number for the first Monday-through-Friday date not in the selected holiday set.",
    keywords: &["fbusdate", "first business date", "month", "holiday"],
    related: &["lbusdate", "isbusday"],
    sections: &[BuiltinDocumentationSection { heading: "Inputs and limits", paragraphs: &[
        "Year and Month can be scalar or compatible numeric arrays. A third argument replaces the default computed holidays; the fourth supplies a seven-element Sunday-first Weekend mask. Results preserve the broadcast input shape.",
        "The fifth `outputType` argument selects `datenum` or `datetime`. Without it, a datetime Holiday argument selects datetime output; otherwise the result uses double serial dates. The default computed holiday list is not a complete historical exchange closure record.",
    ] }, EXECUTION[0]],
    examples: &[BuiltinExample {
        id: "first-business-date", title: "Find the first January trading date",
        program: "d = fbusdate(2024,1);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(d == datenum(2024,1,2));" },
    }, BuiltinExample {
        id: "custom-weekend", title: "Include Saturday in the trading week",
        program: "d = fbusdate(2003,3,[],[1 0 0 0 0 0 0]);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(d == datenum(2003,3,1));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What class is returned?", answer: "By default, a double serial date number, including when Year and Month are integer arrays. A datetime Holiday or `outputType` of `datetime` selects datetime output." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB fbusdate reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/fbusdate.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["The historical market-calendar dataset remains incomplete."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const LBUSDATE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lbusdate"), slug: Some("lbusdate"),
    summary: "Find the last business date of a month.",
    description: "`lbusdate(Year,Month)` returns a serial date number for the last Monday-through-Friday date not in the selected holiday set.",
    keywords: &["lbusdate", "last business date", "month", "holiday"],
    related: &["fbusdate", "isbusday"],
    sections: &[BuiltinDocumentationSection { heading: "Inputs and limits", paragraphs: &[
        "Year and Month can be scalar or compatible numeric arrays. A third argument replaces the default computed holidays; the fourth supplies a seven-element Sunday-first Weekend mask. Results preserve the broadcast input shape.",
        "The fifth `outputType` argument selects `datenum` or `datetime`. Without it, a datetime Holiday argument selects datetime output; otherwise the result uses double serial dates. The default computed holiday list is not a complete historical exchange closure record.",
    ] }, EXECUTION[0]],
    examples: &[BuiltinExample {
        id: "last-business-date", title: "Find the last June trading date",
        program: "d = lbusdate(2024,6);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(d == datenum(2024,6,28));" },
    }, BuiltinExample {
        id: "datetime-output", title: "Request a datetime result",
        program: "d = lbusdate(2024,6,[],[],\"datetime\");",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isdatetime(d)); assert(datenum(d) == datenum(2024,6,28));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What happens if month-end is a weekend?", answer: "The result moves backward to the most recent nonholiday business date." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB lbusdate reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/lbusdate.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["The historical market-calendar dataset remains incomplete."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const HOLIDAYS: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("holidays"), slug: Some("holidays"),
    summary: "Return nontrading dates from RunMat's computed market calendar.",
    description: "`holidays(StartDate,EndDate)` returns dates from the built-in calendar inside an inclusive range. In RunMat mode, a zero-argument call uses the current local calendar year.",
    keywords: &["holidays", "NYSE", "calendar", "nontrading dates"],
    related: &["isbusday", "busdays"],
    sections: &[BuiltinDocumentationSection { heading: "Calendar scope", paragraphs: &[
        "RunMat computes ordinary market holidays from fixed and weekday rules. It does not yet carry the complete NYSE special-closure history or all rule-change dates; use an explicit holiday vector in the other business-day functions when those dates matter.",
        "With no arguments, RunMat reads the host's local year, so its result changes with time. That local-year behavior and the single-numeric-Year convenience form are RunMat-only extensions and are rejected in MATLAB compatibility mode. MATLAB's no-argument form instead returns a full historical list, which RunMat does not yet provide.",
        "The optional third argument selects an alternate list and filters it to the inclusive date range. A two-date numeric range returns serial date numbers; a datetime range returns a datetime column. The RunMat-only single-Year form also returns a datetime column. The computed calendar is limited to a span of 1,000 years.",
    ] }, EXECUTION[0]],
    examples: &[BuiltinExample {
        id: "new-year", title: "Inspect New Year's Day",
        program: "h = holidays(datenum(2024,1,1),datenum(2024,1,3));",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isnumeric(h)); assert(any(h == datenum(2024,1,1)));" },
    }, BuiltinExample {
        id: "alternate-holidays", title: "Filter an alternate holiday list",
        program: "d = datenum(2024,1,15); h = holidays(datenum(2024,1,1),datenum(2024,1,31),d);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(h,d));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can this calendar reproduce every historic exchange closure?", answer: "Not yet. Supply explicit holiday dates to business-day operations for historical data requiring special closures." }],
    links: &[BuiltinDocumentationLink { label: "MATLAB holidays reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/holidays.html") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["The no-argument result depends on the local clock. The historical market dataset is incomplete."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
