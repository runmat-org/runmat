use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Input and result",
        paragraphs: &[
            "`factor(n)` returns the prime factors of a real, nonnegative integer scalar in ascending order. The result is a row vector with the same numeric class as `n`.",
            "Zero and one each return a one-element row containing the input value. Negative, fractional, nonfinite, complex, logical, nonscalar, sparse, and provider-resident inputs produce a structured invalid-input error.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Exact integer factorization",
        paragraphs: &[
            "All eight fixed-width integer classes are accepted on the host, including exact `int64` and `uint64` values above `flintmax`. RunMat factors the native integer payload without converting it through `double`.",
            "The implementation combines deterministic primality testing with recursive factorization and sorts the resulting factors before constructing the output row.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double",
        title: "Factor a double scalar",
        program: "F = factor(200)",
        display_output: Some("F = [2 2 2 5 5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(F, [2 2 2 5 5]));",
        },
    },
    BuiltinExample {
        id: "uint16",
        title: "Preserve an unsigned integer class",
        program: "F = factor(uint16(138))",
        display_output: Some("F = uint16([2 3 23])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(F, 'uint16')); assert(isequal(F, uint16([2 3 23])));",
        },
    },
    BuiltinExample {
        id: "uint64",
        title: "Factor an exact wide integer",
        program: "n = uint64(4294967291) * uint64(4294967279);\nF = factor(n)",
        display_output: Some("F preserves both exact uint64 factors"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isa(F, 'uint64')); assert(isequal(F, uint64([4294967279 4294967291])));",
        },
    },
    BuiltinExample {
        id: "zero-one",
        title: "Handle zero and one",
        program: "F0 = factor(0);\nF1 = factor(1);",
        display_output: Some("F0 = 0 and F1 = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(F0, 0)); assert(isequal(F1, 1));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does factor preserve the input class?", answer: "Yes. Double, single, and every fixed-width integer class produce a row with the same class." },
    BuiltinDocumentationFaq { question: "Are wide integer values converted to double?", answer: "No. Fixed-width integer inputs are factored from their exact native payload, including uint64 values above `flintmax`." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "isprime",
        target: BuiltinDocumentationLinkTarget::Builtin("isprime"),
    },
    BuiltinDocumentationLink {
        label: "primes",
        target: BuiltinDocumentationLinkTarget::Builtin("primes"),
    },
    BuiltinDocumentationLink {
        label: "Compatible factor reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/factor.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Factor runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/factor/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Domains, output classes, wide values, and invalid inputs", location: "crates/runmat-runtime/src/builtins/math/discrete/factor/tests.rs" }],
    notes: &[],
};
pub(super) const FACTOR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("factor"),
    slug: Some("factor"),
    summary: "Return the prime factors of a nonnegative integer scalar.",
    description: "`factor` returns an ascending, same-class row of prime factors using exact host integer arithmetic.",
    keywords: &["factor", "prime factors", "integer", "number theory", "discrete math"],
    related: &["isprime", "primes", "prod"],
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
