use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Inputs and shape",
        paragraphs: &[
            "`lcm(A,B)` computes the least common multiple of each corresponding pair. Values must be finite, real, positive integers.",
            "The inputs must have the same size unless one is scalar. This is scalar expansion, not general implicit expansion.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and overflow",
        paragraphs: &[
            "Matching non-double inputs preserve their class. An integer input may be paired with a scalar double, and single mixed with double returns single.",
            "RunMat computes the result with checked exact arithmetic and returns a structured overflow error when the output class cannot represent it.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &["Resident GPU inputs are gathered for the host calculation, and the result is returned on the host. Distributed arguments are materialized before evaluation."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "array-and-scalar",
        title: "Compute multiples with scalar expansion",
        program: "A = [5 17; 10 60];\nL = lcm(A,45)",
        display_output: Some("L = [45 765; 90 180]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(L, [45 765; 90 180]));",
        },
    },
    BuiltinExample {
        id: "uint16",
        title: "Preserve an unsigned integer class",
        program: "A = uint16([255 511 15]);\nB = uint16([15 127 1023]);\nL = lcm(A,B)",
        display_output: Some("L = uint16([255 64897 5115])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(L, 'uint16')); assert(isequal(L, uint16([255 64897 5115])));",
        },
    },
    BuiltinExample {
        id: "integer-and-double-scalar",
        title: "Combine an integer array with a double scalar",
        program: "L = lcm(int32([6 10]), 15)",
        display_output: Some("L = int32([30 30])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(L, 'int32')); assert(isequal(L, int32([30 30])));",
        },
    },
    BuiltinExample {
        id: "single",
        title: "Preserve single precision",
        program: "L = lcm(single([3 4]), single([8 6]))",
        display_output: Some("L = single([24 12])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(L, 'single')); assert(isequal(L, single([24 12])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does lcm support implicit expansion?", answer: "It supports scalar expansion. Two nonscalar inputs must have exactly the same size." },
    BuiltinDocumentationFaq { question: "Why are mixed integer classes rejected?", answer: "A fixed-width integer input may pair with the same class or a scalar double; other combinations have no compatible result-class rule." },
    BuiltinDocumentationFaq { question: "What happens when the result is too large?", answer: "RunMat reports an overflow error rather than truncating or wrapping the result." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "gcd",
        target: BuiltinDocumentationLinkTarget::Builtin("gcd"),
    },
    BuiltinDocumentationLink {
        label: "primes",
        target: BuiltinDocumentationLinkTarget::Builtin("primes"),
    },
    BuiltinDocumentationLink {
        label: "Compatible lcm reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/lcm.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "LCM runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/lcm/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Classes, shapes, exact arithmetic, and failures", location: "crates/runmat-runtime/src/builtins/math/discrete/lcm/tests.rs" }],
    notes: &[],
};
pub(super) const LCM_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lcm"),
    slug: Some("lcm"),
    summary: "Compute least common multiples for positive integer inputs.",
    description: "`lcm` computes checked elementwise least common multiples for same-size inputs or a scalar and an array.",
    keywords: &["lcm", "least common multiple", "integer", "number theory", "discrete math"],
    related: &["gcd", "primes", "factor"],
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
