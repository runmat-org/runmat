use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("butter"),
    slug: Some("butter"),
    summary: "Design Butterworth IIR filters.",
    description: "`butter(n,Wn)` designs an order-`n` Butterworth filter. A digital scalar cutoff defaults to lowpass, while a two-element cutoff defaults to bandpass. Filter-type text selects low, high, bandpass, or stop, and `'s'` selects an analog design.",
    keywords: &[
        "butter",
        "Butterworth",
        "IIR",
        "lowpass",
        "highpass",
        "bandpass",
        "bandstop",
        "signal processing",
    ],
    related: &["filter", "freqz", "conv", "fft"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and outputs",
            paragraphs: &[
                "`b = butter(n,Wn)` returns numerator coefficients as a row; `[b,a]` returns transfer-function rows suitable for `filter(b,a,x)` in digital mode. `[z,p,k]` returns zero-pole-gain form. `[A,B,C,D]` returns a controllable-canonical state-space realization.",
                "The effective order doubles for bandpass and bandstop designs. Digital cutoff frequencies must be finite, greater than zero, and less than one, with one normalized to Nyquist. Analog cutoff frequencies must be finite and positive. Two-edge bands require strictly increasing cutoffs.",
                "The documented cascaded-transfer-function (`ctf`) forms introduced in MATLAB R2024b are not implemented.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes, extensions, and residency",
            paragraphs: &[
                "MATLAB-compatible calls use double for order and cutoff and limit order to 500. RunMat mode additionally admits independently gated integer, logical, single, resident, option-alias, and larger-order forms up to order 1024.",
                "Typed integer order is parsed exactly as a structural control. Typed integer cutoff is explicitly converted at the double design boundary. Outputs are host double values or complex double tensors according to the designed coefficients and requested representation.",
                "Filter design is host-side. MATLAB-compatible mode rejects resident arguments before gather; RunMat extensions may gather them non-destructively through their owner. Downstream `filter` can use a provider's `iir_filter` hook for supported resident signals; `butter` itself claims no GPU design kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "lowpass",
            title: "Low-pass filter a noisy signal",
            program: "fs = 1000;\nt = (0:999) / fs;\nx = sin(2*pi*25*t) + 0.2*sin(2*pi*200*t);\n[b, a] = butter(4, 80/(fs/2), 'low');\ny = filter(b, a, x);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(b),[1 5])); assert(isequal(size(a),[1 5])); assert(isequal(size(y),size(x))); assert(all(isfinite(y)));",
            },
        },
        BuiltinExample {
            id: "highpass",
            title: "High-pass filter coefficients",
            program: "[b, a] = butter(2, 0.25, 'high');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(b),[1 3])); assert(isequal(size(a),[1 3])); assert(abs(sum(b))<1e-10);",
            },
        },
        BuiltinExample {
            id: "bandpass",
            title: "Band-pass filter coefficients",
            program: "[b, a] = butter(3, [0.2 0.4], 'bandpass');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(b)==7); assert(numel(a)==7); assert(all(isfinite(b)));",
            },
        },
        BuiltinExample {
            id: "analog",
            title: "Analog low-pass prototype",
            program: "[b, a] = butter(1, 2, 's');\nfprintf('%.0f %.0f; %.0f %.0f\\n',b(1),b(2),a(1),a(2));",
            display_output: Some("0 2; 1 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(b),[1 2])); assert(isequal(size(a),[1 2])); assert(max(abs(b-[0 2]))<1e-9); assert(max(abs(a-[1 2]))<1e-9);",
            },
        },
        BuiltinExample {
            id: "zpk",
            title: "Request zero-pole-gain outputs",
            program: "[z, p, k] = butter(2,0.25);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(z)==2); assert(numel(p)==2); assert(isscalar(k));",
            },
        },
        BuiltinExample {
            id: "state-space",
            title: "Request state-space outputs",
            program: "[A, B, C, D] = butter(2,0.25);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(A),[2 2])); assert(isequal(size(B),[2 1])); assert(isequal(size(C),[1 2])); assert(isscalar(D));",
            },
        },
        BuiltinExample {
            id: "invalid-cutoff",
            title: "Reject a cutoff outside the digital band",
            program: "butter(2,1.2)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "MATLAB:butter:InvalidFrequency",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does Wn mean for digital filters?",
            answer: "It is normalized to Nyquist, so a 600 Hz cutoff at 4800 Hz sample rate uses `600/(4800/2)`.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use the result with filter?",
            answer: "Yes. Digital `[b,a]` outputs are row vectors accepted by `filter(b,a,x)`.",
        },
        BuiltinDocumentationFaq {
            question: "Does butter execute on the GPU?",
            answer: "No. Design runs on the host; admitted resident controls are gathered, and downstream filtering may use a separate provider hook.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "filter",
            target: BuiltinDocumentationLinkTarget::Builtin("filter"),
        },
        BuiltinDocumentationLink {
            label: "conv",
            target: BuiltinDocumentationLinkTarget::Builtin("conv"),
        },
        BuiltinDocumentationLink {
            label: "fft",
            target: BuiltinDocumentationLinkTarget::Builtin("fft"),
        },
        BuiltinDocumentationLink {
            label: "Runtime butter implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/butter.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime butter implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/butter.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Butterworth output forms and errors",
                location: "builtins::math::signal::butter::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Butterworth filter workflow",
                location: "bytecode::compile::tests::compile_interprets_butter_filter_workflow",
            },
        ],
        notes: &["Coefficient design and downstream filtering have distinct accelerator policies."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
