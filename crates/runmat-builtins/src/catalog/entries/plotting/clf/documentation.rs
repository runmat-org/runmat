use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("clf"),
    summary: "Clear figure contents in MATLAB and RunMat plotting.",
    description: "`clf` clears the contents of the current figure while leaving the figure handle itself alive. It accepts explicit figure numbers, figure-number arrays, and RunMat's `'all'` selector. Typed integer figure numbers are an exact structural convenience available in RunMat mode; compatibility mode accepts ordinary double figure numbers.",
    slug: Some("clf"),
    keywords: &[
        "clf",
        "clear figure",
        "plotting",
        "graphics handles",
    ],
    related: &[
        "figure",
        "gcf",
        "cla",
        "hold",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `clf` delete the figure handle itself?",
            answer: "No. It clears the figure contents but keeps the figure handle selected and available.",
        },
        BuiltinDocumentationFaq {
            question: "What does `clf('reset')` do today?",
            answer: "RunMat parses the keyword for compatibility, but the current implementation treats it the same as plain `clf`.",
        },
        BuiltinDocumentationFaq {
            question: "What does `clf('all')` return?",
            answer: "It returns the number of figures that were cleared. When no figures exist, it returns `0`.",
        },
        BuiltinDocumentationFaq {
            question: "Can figure numbers use integer classes?",
            answer: "Yes in RunMat mode. `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, and `uint64` are exact structural selectors, subject to the positive `uint32` figure-number range. Compatibility mode requires an ordinary double figure number.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "gcf", target: BuiltinDocumentationLinkTarget::Builtin("gcf") },
        BuiltinDocumentationLink { label: "cla", target: BuiltinDocumentationLinkTarget::Builtin("cla") },
        BuiltinDocumentationLink { label: "hold", target: BuiltinDocumentationLinkTarget::Builtin("hold") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/clf.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/clf.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::clf::tests", location: "builtins::plotting::clf::tests" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
