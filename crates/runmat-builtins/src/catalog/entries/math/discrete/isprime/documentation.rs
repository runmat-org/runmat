use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Elementwise primality testing",
        paragraphs: &[
            "`isprime(X)` returns a logical array with the same shape as `X`. An element is true when the corresponding input is prime and false for zero, one, and composite values.",
            "Input must be a dense, real array of finite, nonnegative integer values. Negative, fractional, nonfinite, complex, logical, sparse, and provider-resident inputs produce a structured invalid-input error.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Exact fixed-width inputs",
        paragraphs: &[
            "All eight fixed-width integer classes are tested from their native payload. Exact `int64` and `uint64` values are not routed through `double`, so values above `flintmax` retain their identity.",
            "The logical output preserves scalar, empty, vector, matrix, and N-D input shapes.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-array",
        title: "Test a row of integer-valued doubles",
        program: "TF = isprime([2 3 0 6 10])",
        display_output: Some("TF = logical([1 1 0 0 0])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(TF, logical([1 1 0 0 0])));",
        },
    },
    BuiltinExample {
        id: "uint16",
        title: "Test unsigned integer input",
        program: "TF = isprime(uint16([333 71 99]))",
        display_output: Some("TF = logical([0 1 0])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(TF)); assert(isequal(TF, logical([0 1 0])));",
        },
    },
    BuiltinExample {
        id: "wide-prime",
        title: "Test an exact uint64 value",
        program: "TF = isprime(0xFFFFFFFFFFFFFFC5u64)",
        display_output: Some("TF = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(TF);",
        },
    },
    BuiltinExample {
        id: "shape",
        title: "Preserve matrix shape",
        program: "X = [2 4; 5 9];\nTF = isprime(X)",
        display_output: Some("TF = logical([1 0; 1 0])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(TF), [2 2])); assert(isequal(TF, logical([1 0; 1 0])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does isprime return for zero and one?", answer: "Both are nonprime, so their output elements are false." },
    BuiltinDocumentationFaq { question: "Does isprime preserve shape?", answer: "Yes. Its logical result has the same shape as the input, including empty and N-D arrays." },
    BuiltinDocumentationFaq { question: "Are uint64 values exact?", answer: "Yes. Fixed-width integer payloads are tested directly without a double conversion." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "factor",
        target: BuiltinDocumentationLinkTarget::Builtin("factor"),
    },
    BuiltinDocumentationLink {
        label: "primes",
        target: BuiltinDocumentationLinkTarget::Builtin("primes"),
    },
    BuiltinDocumentationLink {
        label: "Compatible isprime reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/isprime.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Isprime runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/isprime/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric classes, wide values, shapes, empties, and invalid inputs", location: "crates/runmat-runtime/src/builtins/math/discrete/isprime/tests.rs" }],
    notes: &[],
};
pub(super) const ISPRIME_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isprime"),
    slug: Some("isprime"),
    summary: "Determine which elements of a numeric array are prime.",
    description: "`isprime` tests finite nonnegative integer values exactly and returns a shape-preserving logical array.",
    keywords: &["isprime", "prime", "predicate", "integer", "number theory", "logical"],
    related: &["factor", "primes"],
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
