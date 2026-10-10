use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fir1"),
    slug: Some("fir1"),
    summary: "Design Hamming-windowed or custom-window FIR filters.",
    description: "`fir1(n,Wn)` creates host double row-vector FIR coefficients of order `n`. Scalar Wn defaults to lowpass and a two-edge Wn defaults to bandpass. Filter-type, window, and scale options select other admitted forms.",
    keywords: &[
        "fir1",
        "FIR",
        "windowed sinc",
        "lowpass",
        "highpass",
        "bandpass",
        "bandstop",
    ],
    related: &["freqz", "filter", "filtfilt", "hamming"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and numerical behavior",
            paragraphs: &[
                "`Wn` follows normalized digital frequency, with 1 at Nyquist. A scalar specifies lowpass or highpass; an increasing two-element vector specifies bandpass or stop-band.",
                "The optional window vector must have one coefficient per output tap. The default is symmetric Hamming. `scale` normalizes the passband gain; `noscale` leaves the raw windowed response.",
                "Odd requested orders are raised to the next even order for highpass and bandstop designs. The returned row therefore has `n+2` taps in those cases; otherwise it has `n+1`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and residency",
            paragraphs: &[
                "The documented real order and cutoff forms use double. Independently gated RunMat extensions admit typed integer, logical, and single order, cutoff, or window roles. Typed order is structural and exact; integer cutoff/window values must be exactly representable at the binary64 design boundary.",
                "Resident numeric arguments are gathered non-destructively through their owner for host filter design. Coefficients are currently returned as a host double row, including when inputs were resident; no FIR-design device hook or fused kernel is claimed.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "lowpass",
            title: "Design a low-pass FIR filter",
            program: "b = fir1(4, 0.5);\nfprintf('%d %.4f\\n', length(b), sum(b));",
            display_output: Some("5 1.0000"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(b),[1 5])); assert(abs(sum(b)-1)<1e-9);",
            },
        },
        BuiltinExample {
            id: "response",
            title: "Prepare a frequency-response plot",
            program: "fs = 1000;\nfc = 100;\nN = 51;\nwn = fc / (fs/2);\nb = fir1(N-1, wn, 'low');\n[H, w] = freqz(b, 1, 512, fs);\nfprintf('%d %.1f %.1f\\n', length(b), w(1), w(end));",
            display_output: Some("51 0.0 499.0"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(b)==51); assert(abs(w(1))<1e-12); assert(abs(w(end)-499.0234375)<1e-9);",
            },
        },
        BuiltinExample {
            id: "rectangular",
            title: "Use a rectangular custom window without scaling",
            program: "w = ones(1, 11);\nb = fir1(10, 0.3, w, 'noscale');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(b),[1 11])); assert(all(isfinite(b)));",
            },
        },
        BuiltinExample {
            id: "invalid-cutoff",
            title: "Reject an out-of-band cutoff",
            program: "fir1(4, 1.2)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "MATLAB:fir1:InvalidFrequency",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What window does fir1 use by default?",
            answer: "A symmetric Hamming window.",
        },
        BuiltinDocumentationFaq {
            question: "Can the coefficients be used with filtfilt?",
            answer: "Yes. `filtfilt(b,1,x)` performs zero-phase FIR filtering.",
        },
        BuiltinDocumentationFaq {
            question: "Does fir1 run on the GPU?",
            answer: "The current design executes on the host. Resident input controls can be gathered through their owner, but the output is a host double row.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "filtfilt",
            target: BuiltinDocumentationLinkTarget::Builtin("filtfilt"),
        },
        BuiltinDocumentationLink {
            label: "freqz",
            target: BuiltinDocumentationLinkTarget::Builtin("freqz"),
        },
        BuiltinDocumentationLink {
            label: "hamming",
            target: BuiltinDocumentationLinkTarget::Builtin("hamming"),
        },
        BuiltinDocumentationLink {
            label: "Runtime fir1 implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/fir1.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime fir1 implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/fir1.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "FIR design and class/error tests",
            location: "builtins::math::signal::fir1::tests",
        }],
        notes: &[
            "Host design is intentional; hardware execution of downstream filter is a separate identity.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
