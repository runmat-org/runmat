use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("buttord"),
    slug: Some("buttord"),
    summary: "Select the minimum Butterworth filter order and natural cutoff.",
    description: "`[n,Wn] = buttord(Wp,Ws,Rp,Rs)` finds the minimum Butterworth order meeting passband ripple `Rp` and stopband attenuation `Rs`. Digital edges are normalized so 1 is Nyquist; a final `'s'` selects analog rad/s edges.",
    keywords: &[
        "buttord",
        "Butterworth",
        "filter order",
        "IIR",
        "signal processing",
    ],
    related: &["butter", "cheb2ord", "freqz", "filter"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Response forms",
            paragraphs: &[
                "Scalar passband and stopband edges select lowpass or highpass according to their ordering. Two-element edge pairs select bandpass or bandstop when their intervals are properly nested.",
                "The returned `n` is a host double scalar with an integral value. `Wn` is a host double scalar for scalar edges or a two-element row for edge pairs. Pass both to `butter`; supply its matching highpass, bandpass, or stop selector as needed.",
                "`Rp` and `Rs` are positive finite decibel values with `Rs > Rp`. Digital edges must be finite, positive, and less than one. Analog edges must be finite and positive; RunMat currently does not admit infinite analog edges.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and placement",
            paragraphs: &[
                "Documented frequency and attenuation inputs are real single or double. Independently gated RunMat extensions admit typed-integer frequency and attenuation roles, logical roles, zero-imaginary complex frequency edges, and resident arguments. Integer values cross an explicit binary64 analysis boundary and must be exactly representable.",
                "Order selection is host analysis and a fusion boundary. MATLAB-compatible mode rejects resident inputs before gathering; RunMat extension mode gathers each resident design argument non-destructively and returns host double results. A resident input does not imply a GPU order-selection kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "lowpass",
            title: "Design a lowpass Butterworth filter",
            program: "[n,Wn] = buttord(0.2,0.3,1,40); [b,a] = butter(n,Wn)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 12); assert(abs(Wn-0.2108) < 5e-4);",
            },
        },
        BuiltinExample {
            id: "highpass",
            title: "Design a highpass Butterworth filter",
            program: "[n,Wn] = buttord(0.5,0.3,1,40); [b,a] = butter(n,Wn,'high')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 8); assert(abs(Wn-0.4732) < 5e-4);",
            },
        },
        BuiltinExample {
            id: "analog",
            title: "Select an analog order",
            program: "[n,Wn] = buttord(10,20,1,40,'s'); [b,a] = butter(n,Wn,'s')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 8); assert(Wn > 10 && Wn < 12);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What should I pass to `butter`?",
            answer: "Use both returned `n` and `Wn`, with a highpass or band selector when the chosen edge layout requires it.",
        },
        BuiltinDocumentationFaq {
            question: "Does `buttord` run on the GPU?",
            answer: "No order-selection kernel is claimed. Resident arguments are rejected in MATLAB-compatible mode and can be gathered for host analysis in RunMat extension mode.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "butter",
            target: BuiltinDocumentationLinkTarget::Builtin("butter"),
        },
        BuiltinDocumentationLink {
            label: "freqz",
            target: BuiltinDocumentationLinkTarget::Builtin("freqz"),
        },
        BuiltinDocumentationLink {
            label: "filter",
            target: BuiltinDocumentationLinkTarget::Builtin("filter"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/buttord.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/buttord.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Order and compatibility",
            location: "builtins::math::signal::buttord::tests",
        }],
        notes: &[
            "Resident inputs require an owner and RunMat extension mode; no interactive GPU overload is claimed.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
