use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`strncmpi(A, B, N)` compares at most N leading Unicode scalar values without letter case. String arrays, character-matrix rows, and cell text containers expand over compatible element shapes.",
        "The current RunMat comparison requires both nonmissing text elements to contain at least N characters. Unlike `strncmp`, two matching shorter strings return false when N is longer. For N=0, nonmissing elements compare true; missing elements remain false. This records current RunMat behavior, not a claim of verified MATLAB parity.",
        "N accepts scalar real double, single, logical, and all eight integer classes. Negative integral N acts as zero; nonintegral, nonfinite, or unrepresentable counts reject without rounding. Direct numeric or resident A/B returns scalar false before N validation or provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Comparison and logical result construction are host operations in native and browser runtimes. A supported resident N is gathered and may suspend; there is no GPU text kernel or fusion form. Invalid text, count, or shape can throw.",
        "At most one output is supported. The binding has no callback, workspace, filesystem, network, randomness, or time effect; distributed resident-input execution is unsupported.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Compare a case-insensitive prefix",
        program: "tf = strncmpi(\"RunMat\", \"runway\", 3)",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "short-prefix-runmat",
        title: "A short input does not satisfy a longer count",
        program: "tf = strncmpi(\"ab\", \"AB\", 3)",
        display_output: Some("tf = logical 0"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, false));",
        },
    },
    BuiltinExample {
        id: "exact-integer-count-runmat",
        title: "Use an exact integer prefix count",
        program: "tf = strncmpi(\"RunMat\", \"runway\", uint64(3))",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("strncmpi"), slug: Some("strncmpi"),
    summary: "Compare text inputs case-insensitively up to N leading characters.",
    description: "`strncmpi(A, B, N)` returns logical prefix comparisons with case ignored. Current RunMat short-prefix and missing-value behavior is specified in the behavior section.",
    keywords: &["strncmpi", "strings", "core", "matlab", "compatibility"],
    related: &["string", "char", "strncmp", "strcmpi"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What happens when both strings are shorter than N?", answer: "Current RunMat behavior returns false even when those strings match ignoring case. This differs from `strncmp` and is not asserted as MATLAB parity." },
        BuiltinDocumentationFaq { question: "Does N retain exact integer values?", answer: "Yes. Native integer values are read without a double conversion; counts outside the host range reject." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strncmpi.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strncmpi.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "prefix and count behavior", location: "builtins::strings::core::compat::tests" }],
        notes: &["Legacy sidecar signature, summary, categories, host placement, example, two related links, and source link are retained; an exact-integer example was added. Its zero/negative-count blanket-true claim was narrowed because missing inputs remain false. Short-input semantics remain an explicit unresolved parity decision."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
