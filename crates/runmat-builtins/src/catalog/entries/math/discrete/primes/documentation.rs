use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Prime sequence",
        paragraphs: &[
            "`primes(n)` returns a `1`-by-`N` row containing every prime number less than or equal to the finite real integer scalar `n`. Values below two return an empty `1`-by-`0` row.",
            "The output keeps the input class for double, single, and all eight fixed-width integer classes. Integer inputs are read and returned without conversion through double.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Bounded host execution",
        paragraphs: &[
            "RunMat constructs the sequence with a host sieve. A bounded safety limit is checked before allocation so an accidental extreme bound cannot request unbounded memory.",
            "A resident scalar is gathered before evaluation, and the variable-length result is returned on the host. Distributed inputs are materialized first.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "through-25",
        title: "List primes through 25",
        program: "p = primes(25)",
        display_output: Some("p = [2 3 5 7 11 13 17 19 23]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(p, [2 3 5 7 11 13 17 19 23]));",
        },
    },
    BuiltinExample {
        id: "uint16",
        title: "Preserve an unsigned integer class",
        program: "p = primes(uint16(12))",
        display_output: Some("p = uint16([2 3 5 7 11])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(p, 'uint16')); assert(isequal(p, uint16([2 3 5 7 11])));",
        },
    },
    BuiltinExample {
        id: "signed-class",
        title: "Preserve a signed integer class",
        program: "p = primes(int32(7))",
        display_output: Some("p = int32([2 3 5 7])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(p, 'int32')); assert(isequal(p, int32([2 3 5 7])));",
        },
    },
    BuiltinExample {
        id: "empty-row",
        title: "Return an empty row below two",
        program: "p = primes(1)",
        display_output: Some("p = 1x0 empty double row"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(p, 'double')); assert(isequal(size(p), [1 0]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does primes accept arrays?", answer: "No. Its upper bound must be scalar." },
    BuiltinDocumentationFaq { question: "Does primes preserve fixed-width integer classes?", answer: "Yes. Every signed and unsigned fixed-width integer class produces an exact row of the same class." },
    BuiltinDocumentationFaq { question: "Why can a very large bound be rejected?", answer: "The host sieve has a checked allocation limit that prevents unbounded memory requests." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "isprime",
        target: BuiltinDocumentationLinkTarget::Builtin("isprime"),
    },
    BuiltinDocumentationLink {
        label: "factor",
        target: BuiltinDocumentationLinkTarget::Builtin("factor"),
    },
    BuiltinDocumentationLink {
        label: "Compatible primes reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/primes.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Primes runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/primes/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bounds, classes, empty shapes, and failures", location: "crates/runmat-runtime/src/builtins/math/discrete/primes/tests.rs" }],
    notes: &[],
};
pub(super) const PRIMES_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("primes"),
    slug: Some("primes"),
    summary: "Return prime numbers less than or equal to a scalar bound.",
    description: "`primes` returns a same-class row generated by a bounded exact host sieve.",
    keywords: &["primes", "prime", "number theory", "discrete math", "sieve"],
    related: &["isprime", "factor", "gcd", "lcm"],
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
