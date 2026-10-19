use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/pade.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pade"),
    slug: Some("pade"),
    summary: "Padé numerator and denominator coefficients for one continuous-time pure delay.",
    description: "`[num,den] = pade(T,N)` approximates `exp(-T*s)` by an order-N/order-N rational transfer function. Coefficients are real row vectors in descending powers of s, with a monic denominator. A zero delay returns unit gain.",
    keywords: &["pade", "time delay", "control", "rational approximation"],
    related: &["tf", "freqresp"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported coefficient form",
            paragraphs: &[
                "T is a finite nonnegative real host Single or Double scalar, N is a positive integer host scalar (floating or typed integer), and exactly two outputs are required. Output coefficients follow T's floating class. If a normalized coefficient is not representable in that class, the call fails rather than publishing an infinity.",
                "For T = 0, both coefficient rows are [1] regardless of N. The computation checks cancellation while filling long rows and admits the physical tensor and two-output owners under a managed host context.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Current boundaries",
            paragraphs: &[
                "The no-output plot form is not implemented. `pade(sys,N)` and separate input/output/internal delay orders are not implemented because current ordinary tf/ss models do not retain the necessary delay channels. Discrete-time model approximation is not claimed.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "second-order-delay",
        title: "Approximate a 1.8-second delay",
        program: "[num,den] = pade(1.8,2);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(num-[1 -10/3 100/27])) < 1e-10); assert(max(abs(den-[1 10/3 100/27])) < 1e-10);",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does this approximate delays stored in a transfer-function model?",
        answer: "Not yet. The supported form takes a scalar delay and returns two coefficient rows; model-delay and plotting forms are explicit unsupported boundaries.",
    }],
    links: &[SOURCE, BuiltinDocumentationLink {
        label: "tf",
        target: BuiltinDocumentationLinkTarget::Builtin("tf"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "First/second-order and zero-delay closed forms",
            location: "crates/runmat-runtime/src/builtins/control/pade/tests.rs",
        }],
        notes: &["The model-delay and plotting overloads, and linked-product execution qualification, remain open."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
