use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Display modes",
        paragraphs: &[
            "`format` changes numeric presentation for the current execution thread; it does not change stored values, numerical computation, GPU residency, or `fprintf`/`sprintf` explicit format strings. No argument resets to `short`.",
            "`short` uses four fixed decimals for values in [0.001, 10000), otherwise scientific notation. `long` uses fifteen fixed decimals in that range. `shortE` and `longE` always use scientific notation with four and fourteen decimals; `shortG` and `longG` select compact notation with five and fifteen significant digits.",
            "`rat` approximates small-integer ratios with tolerance 5×10⁻⁷ and displays integers without a denominator; `rational` is an accepted alias. `hex` shows the IEEE 754 double-precision bit pattern. Mode spellings are case-insensitive.",
            "`compact` and `loose` are accepted compatibility keywords but currently leave spacing and the active numeric mode unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "State and placement",
        paragraphs: &[
            "The setting persists across cells on the same execution thread until another `format` call changes it. `format` has no public output; a requested output is rejected before the setting changes. `disp` and automatic output use the active mode; `fprintf` and `sprintf` do not.",
            "This is a console-state operation, not a GPU kernel or fusion candidate. The active host renders later values, including values materialized from a provider by the display consumer.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "short",
        title: "Default short format",
        program: "format short\npi",
        display_output: Some("ans = 3.1416"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "long",
        title: "Long precision",
        program: "format long\npi",
        display_output: Some("ans = 3.141592653589793"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "short-e",
        title: "Short scientific notation",
        program: "format shortE\npi",
        display_output: Some("ans = 3.1416e+00"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "long-e",
        title: "Long scientific notation",
        program: "format longE\npi",
        display_output: Some("ans = 3.14159265358979e+00"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "rational",
        title: "Rational approximation",
        program: "format rat\npi",
        display_output: Some("ans = 355/113"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "hex",
        title: "IEEE hexadecimal display",
        program: "format hex\npi",
        display_output: Some("ans = 400921fb54442d18"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "reset",
        title: "Reset the default",
        program: "format\npi",
        display_output: Some("ans = 3.1416"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does format affect computation results?", answer: "No. It changes display only, not stored numeric classes or values; it does not force every internal value to double." },
    BuiltinDocumentationFaq { question: "Is the format setting persistent across cells?", answer: "Yes, on the same execution thread until changed. It is not a transferable distributed state." },
    BuiltinDocumentationFaq { question: "What is the default format?", answer: "`short`, with four fixed decimals in the ordinary magnitude range." },
    BuiltinDocumentationFaq { question: "Does format affect disp?", answer: "Yes. `disp` and automatic output use it; `fprintf` and `sprintf` use their own explicit formats." },
    BuiltinDocumentationFaq { question: "Are mode names case-sensitive?", answer: "No. `SHORT`, `Short`, and `short` select the same mode." },
    BuiltinDocumentationFaq { question: "What does format rat do for integers?", answer: "It displays integers without a denominator; `rational` is an alias for `rat`." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "disp",
        target: BuiltinDocumentationLinkTarget::Builtin("disp"),
    },
    BuiltinDocumentationLink {
        label: "fprintf",
        target: BuiltinDocumentationLinkTarget::Builtin("fprintf"),
    },
    BuiltinDocumentationLink {
        label: "sprintf",
        target: BuiltinDocumentationLinkTarget::Builtin("sprintf"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("format"),
    slug: Some("format"),
    summary: "Set numeric display format for console output in MATLAB and RunMat.",
    description: "`format` controls numeric presentation for the current execution thread. It supports short/long, scientific, compact-significant, rational, and hexadecimal modes; no argument resets to short.",
    keywords: &["format", "display", "precision", "numeric", "short", "long", "scientific", "rational", "hex"],
    related: &["disp", "fprintf", "sprintf"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime format binding", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/format.rs") },
            BuiltinDocumentationLink { label: "Numeric formatter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-value/src/display.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Format modes and persistence", location: "builtins::io::format::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
