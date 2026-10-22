//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`unifrnd(a, b)` returns a scalar double drawn from `Uniform(a, b)`.", "`unifrnd(a, b, n)` returns an `n × n` matrix of samples.", "`unifrnd(a, b, m, n)` returns an `m × n` matrix of samples.", "`unifrnd(a, b, sz)` accepts a size vector and returns an array with shape `sz`.", "`a` and `b` must be scalar single or double values on the compatibility surface.", "Native typed-integer bounds are independently gated RunMat extensions and must be exactly representable before entering binary64 sampling.", "`a` must be less than `b`; `unifrnd` errors if `a >= b`.", "Native typed-integer size controls are a separately gated RunMat extension and are decoded exactly as structural extents.", "The currently implemented scalar-parameter path returns double output. Complete documented single-parameter class propagation and array parameter broadcasting remain implementation gaps."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["The current scalar-parameter implementation generates native double samples on the host using the shared host RNG. It does not invoke random_unifrnd or a Philox provider hook; resident parameters and dimensions gather before host generation, and output remains host-resident."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation limits",
        paragraphs: &["The implemented scalar-parameter path returns double. Single-parameter output-class propagation and nonscalar-bound broadcasting remain implementation gaps."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Single sample from Uniform(2, 5)",
        program: "rng(0);\nr = unifrnd(2, 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(r)); assert(r>=2 & r<=5); rng(0); assert(isequal(r,unifrnd(2,5)));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Matrix of random values between -1 and 1",
        program: "rng(0);\nX = unifrnd(-1, 1, 2, 3)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(X),[2 3])); assert(all(X(:)>=-1 & X(:)<=1));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Monte Carlo samples for uncertain input bounds",
        program: "rng(0);\nlow = 10; high = 20;\nsamples = unifrnd(low, high, 1, 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(samples),[1 5])); assert(all(samples>=low & samples<=high));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Specifying dimensions with a size vector",
        program: "rng(0);\nT = unifrnd(0, 100, [2 3])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(T),[2 3])); assert(all(T(:)>=0 & T(:)<=100));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Read integer bounds and dimensions structurally",
        program: "rng(0); R = unifrnd(int16(-2),uint8(3),uint8([2 3])); rng(0); expected = unifrnd(-2,3,[2 3]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R,'double')); assert(isequal(size(R),[2 3])); assert(all(R(:)>=-2 & R(:)<=3)); assert(isequal(R,expected));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unifrnd"),
    slug: Some("unifrnd"),
    summary: "Generate uniform random samples on interval [a, b) with MATLAB-compatible size handling.",
    description: "`unifrnd(a, b, ...)` draws native double samples on the host from scalar real bounds a and b over [a, b). Optional scalar and vector size forms choose the output shape. Complete single-parameter class propagation and array-bound broadcasting are not implemented.",
    keywords: &["unifrnd", "uniform", "random", "distribution", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Are the bounds inclusive?",
            answer: "RunMat generates samples on the half-open interval `[a, b)`: values can be equal to `a`, but are always strictly less than `b`.",
        },
        BuiltinDocumentationFaq {
            question: "What is the workaround if unifrnd is unavailable?",
            answer: "`unifrnd(a, b)` is mathematically equivalent to `a + (b - a) * rand()`. For arrays, use `a + (b - a) * rand(sz)` with the same requested size.",
        },
        BuiltinDocumentationFaq {
            question: "Why must a be less than b?",
            answer: "A continuous uniform distribution needs a positive-width interval. `unifrnd` raises an error if `a >= b` to catch invalid bounds early.",
        },
        BuiltinDocumentationFaq {
            question: "What use cases does unifrnd support?",
            answer: "Common uses include Monte Carlo sampling, randomized initial conditions, parameter sweeps over bounded ranges, simulation inputs, and continuous random perturbations.",
        },
        BuiltinDocumentationFaq {
            question: "Does unifrnd fuse with other operations?",
            answer: "No. Random generation is excluded from fusion planning to preserve statistical properties.",
        },
        BuiltinDocumentationFaq {
            question: "How do I control reproducibility?",
            answer: "Use `rng` before calling `unifrnd` to seed the global generator.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "exprnd",
            target: BuiltinDocumentationLinkTarget::Builtin("exprnd"),
        },
        BuiltinDocumentationLink {
            label: "normrnd",
            target: BuiltinDocumentationLinkTarget::Builtin("normrnd"),
        },
        BuiltinDocumentationLink {
            label: "rand",
            target: BuiltinDocumentationLinkTarget::Builtin("rand"),
        },
        BuiltinDocumentationLink {
            label: "randn",
            target: BuiltinDocumentationLinkTarget::Builtin("randn"),
        },
        BuiltinDocumentationLink {
            label: "rng",
            target: BuiltinDocumentationLinkTarget::Builtin("rng"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/unifrnd.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/unifrnd.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::stats::random::unifrnd::tests",
        }],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
