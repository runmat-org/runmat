use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cheb2ord"),
    slug: Some("cheb2ord"),
    summary: "Select the minimum Chebyshev Type II filter order and critical cutoff.",
    description: "`[n,Wn] = cheb2ord(Wp,Ws,Rp,Rs)` finds the minimum Chebyshev Type II order meeting passband ripple `Rp` and stopband attenuation `Rs`. Digital edges are normalized so 1 is Nyquist; a final `'s'` selects analog rad/s edges.",
    keywords: &["cheb2ord", "Chebyshev", "Type II", "filter order", "IIR"],
    related: &["buttord", "freqz", "filter"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Response forms",
            paragraphs: &[
                "Scalar passband and stopband edges select lowpass or highpass. Two-element edge pairs select bandpass or bandstop when their intervals are properly nested.",
                "The returned `n` is a host double scalar with an integral value. `Wn` is a host double scalar for scalar edges or a two-element row for edge pairs. `Wn` is the Type II critical frequency, not simply the original passband edge.",
                "`Rp` and `Rs` are positive finite decibel values with `Rs > Rp`. Digital edges must be finite, positive, and less than one; analog edges must be finite and positive.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and placement",
            paragraphs: &[
                "Documented input values are real single or double. RunMat extension mode additionally admits independently gated typed-integer frequency and attenuation roles, logical input, and resident arguments. The shared real-edge decoder also tolerates complex frequency values only when their imaginary component is zero within its numerical tolerance. Integer values must cross the binary64 analysis boundary exactly.",
                "Order selection is host analysis and a fusion boundary. A resident design argument is gathered non-destructively only in RunMat extension mode. Results are always host double values; a resident input does not imply a GPU order-selection kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "lowpass",
            title: "Select a Type II lowpass order",
            program: "[n,Wn] = cheb2ord(0.2,0.3,1,40)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 6); assert(Wn > 0.2 && Wn < 0.3);",
            },
        },
        BuiltinExample {
            id: "highpass",
            title: "Select a Type II highpass order",
            program: "[n,Wn] = cheb2ord(0.5,0.3,1,40)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 5); assert(Wn > 0.3 && Wn < 0.5);",
            },
        },
        BuiltinExample {
            id: "analog",
            title: "Select an analog Type II order",
            program: "[n,Wn] = cheb2ord(10,20,1,40,'s')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 5); assert(Wn > 10 && Wn < 20);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does `Wn` represent?",
            answer: "It is the Chebyshev Type II critical cutoff satisfying the ripple and attenuation constraints; a two-edge design returns a sorted row pair.",
        },
        BuiltinDocumentationFaq {
            question: "Does `cheb2ord` run on the GPU?",
            answer: "No order-selection kernel is claimed. RunMat extension mode can gather resident design arguments for host analysis; results remain host values.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "buttord",
            target: BuiltinDocumentationLinkTarget::Builtin("buttord"),
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
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/cheb2ord.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/cheb2ord.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Order and compatibility",
            location: "builtins::math::signal::cheb2ord::tests",
        }],
        notes: &[
            "Resident inputs require RunMat extension mode; no interactive GPU overload is claimed.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
