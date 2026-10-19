use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Forms and scaling",
        paragraphs: &[
            "`besselj(nu,Z)` computes the first-kind Bessel function for real order `nu` and real or complex argument `Z`. `nu` and `Z` must have identical size or one must be scalar. The optional third input is exactly `0` (ordinary) or `1` (scaled).",
            "`besselj(nu,Z,1)` evaluates the exponentially scaled result exp(-abs(imag(Z))) * J_nu(Z) inside the AMOS algorithm; it does not first compute an overflowed ordinary result. Negative noninteger orders and the complex branch at negative real arguments are supported.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes, arrays, and numerical range",
        paragraphs: &[
            "Dense real single/double `nu` and dense real/complex single/double `Z` are supported. Any single-precision input selects single output; otherwise output is double. Shape is inherited from the nonscalar operand. The host implementation uses a pure-Rust translation of Amos Algorithm 644 and reports typed numerical failures for overflow, total precision loss, and nonconvergence.",
            "This implementation is host-only. It does not claim a GPU or distributed kernel, silently demote a resident value, or accept sparse, fixed-width integer, logical, or symbolic operands. Inputs in those domains need an explicit supported conversion or host transfer.",
            "RunMat propagates NaN inputs to NaN output. The scalar backend rejects infinite order or argument and reports a typed numerical error; it does not claim MATLAB's warning or nonfinite-result behavior for those cases.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "drum-j0",
        title: "Evaluate the first-kind Bessel function",
        program: "J = besselj(0,1)",
        display_output: Some("J = 0.7652"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(J - 0.765197686557967) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "scaled-complex",
        title: "Avoid complex exponential growth",
        program: "Z = 1i; J = besselj(0,Z,1)",
        display_output: Some("J = exp(-1)*J_0(1i)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(J - exp(-1)*besselj(0,Z)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "When do nu and Z expand?", answer: "Only when one is scalar. Two nonscalar arrays must have exactly the same dimensions; ordinary implicit expansion does not apply." },
    BuiltinDocumentationFaq { question: "What does scale=1 do?", answer: "It multiplies J_nu(Z) by exp(-abs(imag(Z))) and is evaluated in a scaled numerical regime to retain finite complex results." },
    BuiltinDocumentationFaq { question: "Why can a real input produce a complex answer?", answer: "For a negative real argument and noninteger order the principal Bessel branch is complex." },
];

const RELATED: &[&str] = &[
    "gamma", "gammaln", "abs", "imag", "real", "single", "double",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "MathWorks besselj",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://www.mathworks.com/help/matlab/ref/besselj.html",
        ),
    },
    BuiltinDocumentationLink {
        label: "NIST Bessel computation",
        target: BuiltinDocumentationLinkTarget::Source("https://dlmf.nist.gov/10.74"),
    },
    BuiltinDocumentationLink {
        label: "AMOS-based numerical backend",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/elgar328/complex-bessel",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "besselj host runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/bessel_functions/besselj/mod.rs") }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Real, fractional, complex scaled, class, shape, and invalid-form contracts",
        location: "crates/runmat-runtime/src/builtins/math/elementwise/bessel_functions/besselj/tests.rs",
    }],
    notes: &["Host numeric surface is implemented; GPU and distributed residency remain unsupported rather than silently demoted."],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("besselj"), slug: Some("besselj"),
    summary: "Evaluate the first-kind Bessel function for real orders and real or complex arguments.",
    description: "`besselj(nu,Z[,scale])` supports scalar expansion, dense single/double arrays, complex arguments, negative real orders, and exponential scaling on the host.",
    keywords: &["besselj", "Bessel", "special function", "complex", "scaled", "AMOS"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};
