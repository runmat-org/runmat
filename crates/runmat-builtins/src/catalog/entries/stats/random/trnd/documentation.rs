//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`nu` must be a positive scalar or a positive array of degrees of freedom. MATLAB-compatible calls use `single` or `double`; typed-integer degrees of freedom are available in `runmat` compatibility mode.", "Scalar `nu` broadcasts over the requested output size.", "Array `nu` requires the requested size to match the shape of `nu`.", "MATLAB-compatible size controls use integer-valued doubles; typed-integer scalar or vector sizes are available in `runmat` compatibility mode.", "`nu = Inf` generates standard normal random numbers."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["Sampling currently runs on the host RNG; GPU parameters use a gather-and-reupload fallback so output remains resident with matching floating precision."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f32. See the distribution, integer-extension, and residency rules above."],
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
        program: "rng(0);\nr = trnd(10)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(r)); assert(isa(r,'double')); assert(isfinite(r)); rng(0); assert(isequal(r,trnd(10)));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Matrix sample",
        program: "rng(0);\nr = trnd(1, 5, 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(r),[5 5])); assert(all(isfinite(r(:))));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Retain array shape and single precision",
        program: "rng(0); R = trnd(single([2 3;4 5]));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R,'single')); assert(isequal(size(R),[2 2])); assert(all(isfinite(R(:))));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trnd"),
    slug: Some("trnd"),
    summary: "Generate Student's t random numbers.",
    description: "`trnd(nu)` generates Student's t random numbers with `nu` degrees of freedom. Size arguments can be supplied as separate dimensions or as a size vector.",
    keywords: &["trnd", "student t", "random", "statistics", "distribution"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "How does RunMat generate trnd samples?",
        answer: "RunMat draws a standard normal variate and scales it by an independent chi-square variate generated through a gamma sampler.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "tcdf",
            target: BuiltinDocumentationLinkTarget::Builtin("tcdf"),
        },
        BuiltinDocumentationLink {
            label: "tpdf",
            target: BuiltinDocumentationLinkTarget::Builtin("tpdf"),
        },
        BuiltinDocumentationLink {
            label: "tinv",
            target: BuiltinDocumentationLinkTarget::Builtin("tinv"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/trnd.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/trnd.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::stats::random::trnd::tests",
        }],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
