use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("figure"),
    summary: "Create or select RunMat plotting figures.",
    description: "`figure` creates a new plotting figure or selects an existing one as current. RunMat currently encodes graphics handles as host `f64` identifiers instead of MATLAB Figure objects; this is a general graphics-representation gap, not numeric MATLAB equivalence. Double numeric selection is the compatibility form, while typed-integer targets, single-precision targets, the `'next'` selector, and typed-integer property values are independently gated RunMat-only extensions.",
    slug: Some("figure"),
    keywords: &[
        "figure",
        "plotting",
        "graphics handle",
        "window selection",
    ],
    related: &[
        "gcf",
        "gca",
        "clf",
        "subplot",
        "plot",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `figure(n)` require the handle to exist already?",
            answer: "No. RunMat selects figure `n`, creating it on demand if necessary.",
        },
        BuiltinDocumentationFaq {
            question: "What does `figure('next')` do?",
            answer: "In RunMat extension mode it forces creation of a new figure rather than reusing an existing handle. Compatibility mode rejects this selector before changing graphics state.",
        },
        BuiltinDocumentationFaq {
            question: "Can `figure` set window properties at creation time?",
            answer: "Yes. Pass alternating property/value pairs, either directly or after a numeric handle: `figure('Name', 'demo')` or `figure(3, 'Name', 'demo', 'NumberTitle', 'off')`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `figure` draw anything by itself?",
            answer: "No. It only manages the active figure selection. Plotting happens when you call builtins like `plot`, `scatter`, or `surf`.",
        },
        BuiltinDocumentationFaq {
            question: "Why is the returned handle a double scalar?",
            answer: "RunMat's current graphics registry uses a host `f64` identifier. MATLAB uses Figure objects, so this representation is tracked as a general graphics-model gap rather than claimed as numeric compatibility.",
        },
        BuiltinDocumentationFaq {
            question: "Can integer, single, logical, or GPU-resident values select a figure?",
            answer: "Host scalar integers and singles are independent RunMat-only extensions. Integers are read from authoritative storage and must be positive `u32` values. Logical and resident numeric targets reject.",
        },
        BuiltinDocumentationFaq {
            question: "Can figure property values use typed integers?",
            answer: "Only in RunMat extension mode. `Position` and graphics-object aliases reject integer values that would round at an `f64` boundary, `Color` accepts only the usual zero/one integer RGB components, and `NumberTitle`/`Visible` use exact zero/nonzero tests. These gates run before figure creation, selection, or property mutation.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "gcf", target: BuiltinDocumentationLinkTarget::Builtin("gcf") },
        BuiltinDocumentationLink { label: "gca", target: BuiltinDocumentationLinkTarget::Builtin("gca") },
        BuiltinDocumentationLink { label: "clf", target: BuiltinDocumentationLinkTarget::Builtin("clf") },
        BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
        BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/figure.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/figure.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::figure::tests", location: "builtins::plotting::figure::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "runmat-plot/tests/renderer_tests.rs", location: "runmat-plot/tests/renderer_tests.rs" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
