use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("freqz"),
    slug: Some("freqz"),
    summary: "Evaluate digital filter frequency response.",
    description: "`freqz(b,a,n)` evaluates a digital transfer function at `n` points from zero to below Nyquist. `[H,w]` returns host complex response and host double frequency columns; `fs` expresses `w` in Hz. The no-output form plots the magnitude and phase response when graphics are available.",
    keywords: &["freqz", "frequency response", "filter", "FIR", "IIR"],
    related: &["butter", "filter", "fir1", "filtfilt"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Frequency grid and results",
            paragraphs: &[
                "The default grid contains 512 points. An explicit numeric `n` must be an integer at least 2; an empty `n` selects the default. The grid excludes the Nyquist endpoint.",
                "Single-precision coefficients produce complex single `H`; otherwise `H` is complex double. `w` is always double. Both numerical outputs are columns, including a two-point response.",
                "Without `fs`, `w` is in radians/sample over `[0,pi)`; with a positive finite `fs`, it is in Hz over `[0,fs/2)`. Output forms do not create figures.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Compatibility and placement",
            paragraphs: &[
                "Typed integer/logical coefficients and non-double controls are separately gated RunMat extensions. Integer coefficient and sample-rate values must be exactly representable at the binary64 evaluation boundary; integer `n` is parsed exactly.",
                "Resident inputs are a RunMat extension. After compatibility preflight they are gathered non-destructively through their owner for host analysis; this identity does not claim a device response kernel or resident output.",
                "The zero-output form plots magnitude and phase on separate axes through RunMat's graphics service. If plotting support is unavailable, that form refuses before reading resident inputs. Numerical output forms need no graphics capability.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "fir",
            title: "Evaluate a simple FIR response",
            program: "[H, w] = freqz([1 1], 1, 4);\nfprintf('%.1f %.1f %.4f\\n', real(H(1)), imag(H(3)), w(3));",
            display_output: Some("2.0 -1.0 1.5708"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(H),[4 1])); assert(isequal(size(w),[4 1])); assert(abs(real(H(1))-2)<1e-12); assert(abs(imag(H(3))+1)<1e-12);",
            },
        },
        BuiltinExample {
            id: "hz",
            title: "Evaluate a response in Hz",
            program: "b = fir1(20, 0.25);\n[H, f] = freqz(b, 1, 8, 1000);\nfprintf('%.1f %.1f\\n', f(1), f(end));",
            display_output: Some("0.0 437.5"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(H),[8 1])); assert(abs(f(1))<1e-12); assert(abs(f(end)-437.5)<1e-12);",
            },
        },
        BuiltinExample {
            id: "iir",
            title: "Inspect a Butterworth low-pass response",
            program: "[b, a] = butter(2, 0.25, 'low');\n[H, w] = freqz(b, a, 8);\nfprintf('%d %.4f %.4f\\n', length(H), abs(H(1)), w(end));",
            display_output: Some("8 1.0000 2.7489"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(H)==8); assert(abs(abs(H(1))-1)<1e-9); assert(abs(w(end)-7*pi/8)<1e-9);",
            },
        },
        BuiltinExample {
            id: "plot",
            title: "Plot magnitude and phase without output variables",
            program: "freqz([1 1],1,8);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(~isempty(findobj(gcf,'Type','line')));",
            },
        },
        BuiltinExample {
            id: "invalid-n",
            title: "Reject an undersized response grid",
            program: "freqz(1,1,1)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "MATLAB:freqz:InvalidN",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does freqz include Nyquist?",
            answer: "No. The `n`-point grid spans `[0,pi)` or `[0,fs/2)`.",
        },
        BuiltinDocumentationFaq {
            question: "Can freqz analyze IIR filters?",
            answer: "Yes. Pass numerator `b` and denominator `a`; the transfer function is evaluated directly on the unit circle.",
        },
        BuiltinDocumentationFaq {
            question: "Are results resident on the GPU?",
            answer: "No. Admitted resident arguments are gathered through their owner, and the response and frequency vectors are host outputs.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "butter",
            target: BuiltinDocumentationLinkTarget::Builtin("butter"),
        },
        BuiltinDocumentationLink {
            label: "filter",
            target: BuiltinDocumentationLinkTarget::Builtin("filter"),
        },
        BuiltinDocumentationLink {
            label: "fir1",
            target: BuiltinDocumentationLinkTarget::Builtin("fir1"),
        },
        BuiltinDocumentationLink {
            label: "filtfilt",
            target: BuiltinDocumentationLinkTarget::Builtin("filtfilt"),
        },
        BuiltinDocumentationLink {
            label: "Runtime freqz implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/freqz.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime freqz implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/freqz.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "frequency grid, precision, error, and plot tests",
            location: "builtins::math::signal::freqz::tests",
        }],
        notes: &[
            "Zero-output plotting is a new supported behavior; numeric response behavior remains unchanged.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
