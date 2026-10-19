use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unwrap"),
    slug: Some("unwrap"),
    summary: "Correct phase-angle jumps by adding multiples of 2*pi.",
    description: "`Q = unwrap(P)` adjusts real phase angles in radians by multiples of `2*pi` whenever adjacent samples jump by more than the tolerance.",
    keywords: &["unwrap", "phase", "angle", "signal processing", "radians"],
    related: &["angle", "hilbert", "diff", "fft"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Tolerance and dimension",
            paragraphs: &[
                "The first nonsingleton dimension is used by default. `unwrap(P,tol)` uses `max(tol,pi)`; `unwrap(P,[],dim)` keeps the default tolerance and selects a structural dimension.",
                "The result has the same shape as P. Single phase data returns single; double returns double. Complex phase data is rejected; use `angle` first.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Compatibility and placement",
            paragraphs: &[
                "Logical and typed-integer phase data and typed-integer tolerance are separately gated RunMat extensions. Phase and tolerance integers must cross an exact binary64 boundary; integer/logical phase results are double. Dimension integers are decoded as exact structural indices.",
                "Resident inputs are downloaded non-destructively through their owning provider. Stateful phase correction currently runs on the host and returns a host tensor; re-upload is explicit. Operational download and cancellation failures are not retried as another route.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "wrapped-vector",
            title: "Correct a wrapped phase vector",
            program: "p = [0 1 2 2-2*pi 3-2*pi]; q = unwrap(p)",
            display_output: Some("q = [0 1 2 2 3]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(q-[0 1 2 2 3])) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "custom-tolerance",
            title: "Use a custom tolerance",
            program: "p = [0 1 1-2*pi]; q = unwrap(p,10)",
            display_output: Some("q = [0 1 -5.2832]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(q-p)) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "matrix-dimension",
            title: "Unwrap across matrix rows",
            program: "P = [0 1 1-2*pi; 0 -1 -1+2*pi]; Q = unwrap(P,[],2)",
            display_output: Some("Q = [0 1 1; 0 -1 -1]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(max(abs(Q-[0 1 1;0 -1 -1]))) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "analytic-phase",
            title: "Recover phase from an analytic signal",
            program: "x = cos(0:0.1:10); z = hilbert(x); phase = unwrap(angle(z));",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(phase),size(x))); assert(all(isfinite(phase)));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What tolerance is used by default?",
            answer: "pi; any smaller nonnegative tolerance is raised to pi.",
        },
        BuiltinDocumentationFaq {
            question: "Which dimension is unwrapped?",
            answer: "The first nonsingleton dimension unless dim is supplied.",
        },
        BuiltinDocumentationFaq {
            question: "Does unwrap change shape or resident placement?",
            answer: "Shape is preserved. The current stateful implementation gathers resident input and returns a host tensor.",
        },
        BuiltinDocumentationFaq {
            question: "Can I pass complex input?",
            answer: "No. Convert complex samples to real angles with angle first.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "angle",
            target: BuiltinDocumentationLinkTarget::Builtin("angle"),
        },
        BuiltinDocumentationLink {
            label: "hilbert",
            target: BuiltinDocumentationLinkTarget::Builtin("hilbert"),
        },
        BuiltinDocumentationLink {
            label: "diff",
            target: BuiltinDocumentationLinkTarget::Builtin("diff"),
        },
        BuiltinDocumentationLink {
            label: "fft",
            target: BuiltinDocumentationLinkTarget::Builtin("fft"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/unwrap.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/unwrap.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Phase, class, dimension, and resident materialization",
            location: "builtins::math::signal::unwrap::tests",
        }],
        notes: &[
            "Portable examples execute the host reference route; provider ownership requires a resident fixture.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
