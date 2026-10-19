use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("envelope"),
    slug: Some("envelope"),
    summary: "Compute upper and lower signal envelopes.",
    description: "`envelope(x)` computes upper and lower bounds for a finite real vector or each column of a finite real matrix. The default analytic method removes each channel mean, takes the analytic-signal magnitude, and restores the mean to both bounds.",
    keywords: &[
        "envelope",
        "analytic signal",
        "Hilbert",
        "RMS",
        "peak envelope",
        "signal processing",
    ],
    related: &["hilbert", "pchip", "plot"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Methods and outputs",
            paragraphs: &[
                "`yupper = envelope(x)` returns the upper bound; `[yupper,ylower] = envelope(x)` returns both. `envelope(x,n)` and `envelope(x,n,'analytic')` use a Kaiser-windowed Hilbert FIR filter of length n.",
                "`envelope(x,wl,'rms')` computes centered sliding RMS bounds; `envelope(x,np,'peak')` selects separated extrema and interpolates them with a spline. Statement form with zero outputs renders the signal and both bounds through the plot pipeline.",
                "Outputs preserve the input shape. RunMat preserves single output for single floating input on host and provider paths; the public compatibility source does not state this class detail.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Compatibility and placement",
            paragraphs: &[
                "Single and double real signals are documented. Typed-integer signal data and typed-integer n are independently gated RunMat extensions. Integer data must cross an exact floating computation boundary; n is an exact positive structural control.",
                "Real resident floating input can use its owning provider's analytic, analytic-FIR, or RMS operation and return distinct resident outputs. Peak and unavailable provider forms materialize through the owner and restore when supported; an error after invoking an admitted provider hook is terminal, not replayed on the host.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "am-analytic",
            title: "Analytic envelope of an AM tone",
            program: "t = (0:127)/128; x = (1.25+0.5*sin(2*pi*t)).*sin(2*pi*8*t); [up,lo] = envelope(x);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(up),size(x))); assert(isequal(size(lo),size(x))); assert(all(up>=lo));",
            },
        },
        BuiltinExample {
            id: "rms",
            title: "Centered moving RMS",
            program: "x = [0 3 4 0]; [up,lo] = envelope(x,3,'rms');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(up),size(x))); assert(max(abs(up+lo)) < 1e-10); assert(up(2)>0);",
            },
        },
        BuiltinExample {
            id: "peak",
            title: "Peak envelope",
            program: "x = [0 2 0 -1 0 3 0]; [up,lo] = envelope(x,2,'peak');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(up),size(x))); assert(isequal(size(lo),size(x))); assert(all(isfinite(up)));",
            },
        },
        BuiltinExample {
            id: "matrix-columns",
            title: "Process each matrix column",
            program: "x = [sin(0:0.1:10)' cos(0:0.1:10)']; [up,lo] = envelope(x);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(up),size(x))); assert(isequal(size(lo),size(x)));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What is the default method?",
            answer: "Analytic-envelope mode using mean-centered analytic-signal magnitude.",
        },
        BuiltinDocumentationFaq {
            question: "How are matrices handled?",
            answer: "Each column is processed independently and both outputs retain the input shape.",
        },
        BuiltinDocumentationFaq {
            question: "Does statement form plot?",
            answer: "Yes, zero-output statement form renders the signal and both envelopes when plotting is available.",
        },
        BuiltinDocumentationFaq {
            question: "Can the outputs stay resident?",
            answer: "An owning provider may execute analytic or RMS forms; peak mode and unsupported forms materialize through the owner. Operational provider errors are terminal.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "hilbert",
            target: BuiltinDocumentationLinkTarget::Builtin("hilbert"),
        },
        BuiltinDocumentationLink {
            label: "pchip",
            target: BuiltinDocumentationLinkTarget::Builtin("pchip"),
        },
        BuiltinDocumentationLink {
            label: "plot",
            target: BuiltinDocumentationLinkTarget::Builtin("plot"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/envelope.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/envelope.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Analytic, RMS, peak, class, and resident output ownership",
            location: "builtins::math::signal::envelope::tests",
        }],
        notes: &[
            "Portable examples cover host numerics; real resident execution needs a capable provider. Plotting requires plot-core.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
