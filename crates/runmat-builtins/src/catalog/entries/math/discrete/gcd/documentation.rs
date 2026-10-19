use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Elementwise divisors",
        paragraphs: &[
            "`gcd(A,B)` computes a nonnegative greatest common divisor for each corresponding pair. The inputs must have the same size unless one is scalar.",
            "Inputs may contain positive, negative, or zero integer values. Matching non-double classes are preserved; an integer array may also be paired with a scalar double.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Bezout coefficients",
        paragraphs: &[
            "`[G,U,V] = gcd(A,B)` also returns coefficients satisfying `A.*U + B.*V = G`. Extended outputs support double, single, and signed integer classes.",
            "Unsigned integer inputs are valid for `G = gcd(A,B)` but cannot represent negative coefficients, so RunMat rejects the extended form for unsigned output classes.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &["Resident GPU inputs are gathered before the exact host calculation. The result is a host value. Distributed inputs are materialized under the same contract."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "array-and-scalar",
        title: "Compute divisors with scalar expansion",
        program: "G = gcd([12 15], 18)",
        display_output: Some("G = [6 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(G, [6 3]));",
        },
    },
    BuiltinExample {
        id: "signed-integer",
        title: "Preserve a signed integer class",
        program: "G = gcd(int16([-30 21]), int16([18 14]))",
        display_output: Some("G = int16([6 7])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(G, 'int16')); assert(isequal(G, int16([6 7])));",
        },
    },
    BuiltinExample {
        id: "bezout",
        title: "Verify the Bezout identity",
        program: "A = 30; B = 18;\n[G,U,V] = gcd(A,B)",
        display_output: Some("A*U + B*V = G"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(G == 6); assert(A.*U + B.*V == G);",
        },
    },
    BuiltinExample {
        id: "zero",
        title: "Compute a divisor with zero",
        program: "G = gcd(0, -42)",
        display_output: Some("G = 42"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(G == 42);",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does gcd use implicit expansion?", answer: "No. It supports scalar expansion; two nonscalar inputs must have exactly the same size." },
    BuiltinDocumentationFaq { question: "Why are extended outputs unavailable for unsigned integers?", answer: "Bezout coefficients may be negative, which an unsigned result class cannot represent." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "lcm",
        target: BuiltinDocumentationLinkTarget::Builtin("lcm"),
    },
    BuiltinDocumentationLink {
        label: "factor",
        target: BuiltinDocumentationLinkTarget::Builtin("factor"),
    },
    BuiltinDocumentationLink {
        label: "Compatible gcd reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/gcd.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "GCD runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/gcd/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Classes, shapes, signs, extended outputs, and failures", location: "crates/runmat-runtime/src/builtins/math/discrete/gcd/tests.rs" }],
    notes: &[],
};
pub(super) const GCD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gcd"),
    slug: Some("gcd"),
    summary: "Compute greatest common divisors for real integer-valued inputs.",
    description: "`gcd` computes elementwise greatest common divisors and can return Bezout coefficients for signed classes.",
    keywords: &["gcd", "greatest common divisor", "Bezout", "integer", "number theory"],
    related: &["lcm", "factor", "isprime"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("Before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
