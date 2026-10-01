use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/finance/macd.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("macd"), slug: Some("macd"),
    summary: "Compute the Moving Average Convergence/Divergence indicator.",
    description: "`macd(Data)` computes the 12-period minus 26-period exponential moving-average line and its 9-period exponential signal line from closing prices.",
    keywords: &["macd", "finance", "technical indicator", "moving average", "ema"],
    related: &["table", "timetable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input forms and output", paragraphs: &[
            "Raw matrix input is M-by-4, with High, Low, Open, and Close prices in columns 1–4. The MATLAB-compatible raw matrix form uses double. In `runmat` compatibility mode, native-single and fixed-width integer matrices are extensions; integer prices must be exactly representable in binary64 before entering the EMA calculation.",
            "Table and timetable input must have numeric M-by-1 High, Low, Open, and Close variables, matched without case sensitivity. Integer variables retain native storage until the checked binary64 EMA boundary. The output container keeps the input row count and table/timetable kind. Its computed Close variable preserves a native-single Close input; integer and double Close inputs produce double data.",
            "One output returns MACDLine; `[MACDLine, SignalLine]` requests both. For a raw matrix, each output is an M-by-1 double column. Empty M-by-4 input yields empty M-by-1 outputs. NaN values propagate through the recurrence.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The recurrence uses 0.15 for the fast EMA, 0.075 for the slow EMA, and 0.20 for the signal EMA. Its first EMA value is the first closing price. The recurrence is sequential along rows.",
            "There is no direct GPU MACD kernel. Resident inputs gather through their owning provider. The host recurrence computes in double and restores single output for tabular inputs with single Close values; raw matrix input produces double output. MACD is a fusion boundary, not an elementwise GPU operation; it does not mutate the input.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "matrix", title: "Compute MACD and signal line",
            program: "priceData = [11 9 10 10; 12 10 11 11; 13 11 12 12]; [m,s] = macd(priceData); fprintf('MACD(3) = %.6f, Signal(3) = %.6f\\n', m(3), s(3));",
            display_output: Some("MACD(3) = 0.208125, Signal(3) = 0.053625"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(m), [3 1])); assert(isequal(size(s), [3 1])); assert(abs(m(1))<1e-12); assert(abs(m(2)-0.075)<1e-12); assert(abs(m(3)-0.208125)<1e-12); assert(abs(s(3)-0.053625)<1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Are resident inputs processed on a GPU?",
        answer: "No. MACD uses a sequential host recurrence and gathers resident inputs first. The two outputs are host values.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") },
        BuiltinDocumentationLink { label: "timetable", target: BuiltinDocumentationLinkTarget::Builtin("timetable") },
        BuiltinDocumentationLink { label: "MathWorks macd documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/macd.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Matrix, table/timetable, integer-boundary, compatibility, and output-count tests", location: "crates/runmat-runtime/src/builtins/finance/macd.rs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
