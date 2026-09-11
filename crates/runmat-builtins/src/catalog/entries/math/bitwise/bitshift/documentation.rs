use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`C = bitshift(A, k)` shifts each integer-valued element of A. Positive counts shift left and discard bits beyond the class width; negative counts shift right. Signed right shifts preserve the sign bit.",
            "A and k use scalar expansion or exactly matching nonscalar sizes. Every fixed-width integer class is accepted for either argument, while A alone determines output class.",
            "`C = bitshift(A, k, assumedtype)` selects the signed or unsigned width for integer-valued `double` A. Unsupported, fractional, nonfinite, complex, or incompatible-shape inputs produce structured errors.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "The documented resident domain requires at least one non-64-bit integer array and excludes signed A. Exact host fallback restores the result to the owning provider.",
            "Single and logical inputs, 64-bit resident integers, signed resident A, resident forms without an integer array, and resident `assumedtype` calls are separately gated RunMat extensions.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "left-right",
        title: "Shift left and right",
        program: "left = bitshift(uint8(3), 2);\nright = bitshift(int8(-8), -2)",
        display_output: Some("left = uint8(12), right = int8(-2)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(left == uint8(12)); assert(right == int8(-2));",
        },
    },
    BuiltinExample {
        id: "broadcast-count",
        title: "Broadcast a shift count",
        program: "a = uint16([1 2 3]);\nc = bitshift(a, 4)",
        display_output: Some("c = uint16([16 32 48])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(c, 'uint16')); assert(isequal(c, uint16([16 32 48])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[BuiltinDocumentationFaq {
    question: "Does a right shift of a signed value keep its sign?",
    answer: "Yes. Negative signed integers use an arithmetic right shift with sign extension.",
}];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Shift runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/bitshift.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Widths, signed shifts, exact shapes, sparse values, compatibility gates, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/math/bitwise/engine/tests",
    }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitshift"),
    slug: Some("bitshift"),
    summary: "Shift integer-valued values left or right by bit counts.",
    description:
        "`bitshift` shifts values within the data input's signed or unsigned integer width.",
    keywords: &["bitshift", "bitwise", "shift", "integer", "gpuArray"],
    related: &["bitand", "bitcmp", "bitget", "bitset"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: &[
        BuiltinDocumentationLink {
            label: "bitand",
            target: BuiltinDocumentationLinkTarget::Builtin("bitand"),
        },
        BuiltinDocumentationLink {
            label: "bitget",
            target: BuiltinDocumentationLinkTarget::Builtin("bitget"),
        },
    ],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
