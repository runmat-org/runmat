//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`n` must contain positive integer-valued single or double values.", "Native typed-integer `n` is a RunMat extension. It is gated before gather and admitted only when every value is exactly representable at the binary64 sampling boundary.", "Scalar `n` broadcasts over the requested output size.", "Array `n` requires the requested size to match the shape of `n`.", "A single size argument creates a square matrix, following MATLAB random-size conventions.", "Documented size controls are single or double. Native typed-integer sizes are a separately gated RunMat extension and are decoded exactly as structural extents."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["GPU inputs are gathered to host for compatibility until provider sampling hooks exist."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Scalar sample",
        program: "rng(0);\nr = unidrnd(10)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(r)); assert(isa(r,'double')); assert(r>=1 & r<=10 & r==floor(r));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Matrix sample",
        program: "rng(0);\nr = unidrnd(6, 3, 4)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(r),[3 4])); assert(all(r(:)>=1 & r(:)<=6 & r(:)==floor(r(:))));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Read native integer limits and shape exactly",
        program: "rng(0); R = unidrnd(uint16(6),uint8([2 3])); rng(0); expected = unidrnd(6,[2 3]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R,'double')); assert(isequal(size(R),[2 3])); assert(all(R(:)>=1 & R(:)<=6 & R(:)==floor(R(:)))); assert(isequal(R,expected));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unidrnd"),
    slug: Some("unidrnd"),
    summary: "Generate random integers from a discrete uniform distribution.",
    description: "`unidrnd(n)` generates random integers from `1` through `n`. Size arguments can be supplied as separate dimensions or as a size vector.",
    keywords: &["unidrnd", "uniform", "discrete", "random", "integer", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "What class does unidrnd return?",
        answer: "RunMat returns double-precision numeric values, matching the common MATLAB numeric result shape.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "randi",
            target: BuiltinDocumentationLinkTarget::Builtin("randi"),
        },
        BuiltinDocumentationLink {
            label: "randsample",
            target: BuiltinDocumentationLinkTarget::Builtin("randsample"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/sampling.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/sampling.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::stats::random::sampling::tests",
        }],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
